"""ComfyUI-Idol: one image -> 3D Gaussian splat, using speridlabs/idol.

Nodes
  Idol Loader      - load/keep the 1.8B pipeline (weights auto-download to models/idol)
  Idol Preprocess  - the 1024^2 image the model sees (background removal + crop)
  Idol Generate    - IMAGE -> SPLAT (+ preprocessed IMAGE + .ply path)
  Idol Unload      - drop the pipeline and free its VRAM

The encoder half (DINOv3 ViT-H/16+ and the FLUX.2 VAE) is loaded from local weights:
DINOv3 from the ungated timm mirror, the VAE from ComfyUI's own models/vae/flux2-vae.safetensors.
"""
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file

import folder_paths

try:
    import comfy.model_management as mm
except Exception:  # standalone import (tests)
    mm = None
try:
    import comfy.utils as comfy_utils
except Exception:
    comfy_utils = None

from .idol import pipeline as idol_pipeline
from .idol.camera import decode_camera
from .idol.conditioning import IMAGENET_MEAN, IMAGENET_STD
from .idol.pipeline import IdolPipeline
from .idol.preprocess import CROP_MARGIN
from .idol.sampler import sample_flow

try:
    from comfy_api.latest import Types
    SPLAT_TYPE = Types.SPLAT
except Exception:
    try:
        from comfy_api.latest._util.geometry_types import SPLAT as SPLAT_TYPE
    except Exception:
        SPLAT_TYPE = None

MODELS_DIR = folder_paths.models_dir
IDOL_DIR = os.path.join(MODELS_DIR, "idol")
DINO_NAME = "dinov3_vith16plus.safetensors"
DINO_REPO = "timm/vit_huge_plus_patch16_dinov3.lvd1689m"
DINO_URL = f"https://huggingface.co/{DINO_REPO}/resolve/main/model.safetensors"
VAE_LOCAL = os.path.join(MODELS_DIR, "vae", "flux2-vae.safetensors")
VAE_URL = "https://huggingface.co/Comfy-Org/Flux-2-Dev/resolve/main/split_files/vae/flux2-vae.safetensors"

# file -> expected byte size (also used as the "is this download complete" check)
IDOL_FILES = {
    "config.json": 1335,
    "denoiser.safetensors": 1482258308,
    "decoder.safetensors": 203171696,
    "decoder_gs.safetensors": 949083036,
    "birefnet.safetensors": 444466856,
}
DINO_BYTES = 3362084520
GAUSSIAN_COUNTS = ["32768", "65536", "131072", "262144", "524288", "1048576"]

# ~7.2 GB peak measured on a 4090; ask ComfyUI to make this much room.
VRAM_REQUEST = 9_000_000_000

_PIPE_CACHE = {}


# --------------------------------------------------------------------------- downloads

def _url_open(url):
    req = urllib.request.Request(url, headers={"User-Agent": "comfyui-idol"})
    try:
        return urllib.request.urlopen(req, timeout=60)
    except (ssl.SSLError, urllib.error.URLError):
        # MITM / missing local CA bundle: retry unverified rather than fail the node
        ctx = ssl._create_unverified_context()
        return urllib.request.urlopen(req, timeout=60, context=ctx)


def _download(url, dest, expected=None):
    if os.path.exists(dest) and (expected is None or os.path.getsize(dest) == expected):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    print(f"[idol] downloading {os.path.basename(dest)} from {url}")
    resp = _url_open(url)
    total = int(resp.headers.get("Content-Length") or 0)
    done = 0
    last = 0.0
    with resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2.0:
                last = time.time()
                pct = f"{100 * done / total:5.1f}%" if total else "  ?  "
                print(f"[idol]   {pct}  {done / 1e9:6.2f} / {total / 1e9:6.2f} GB", end="\r")
    print()
    if expected is not None and done != expected:
        raise RuntimeError(f"[idol] {os.path.basename(dest)}: got {done} bytes, expected {expected}")
    os.replace(tmp, dest)
    return dest


def ensure_weights(download=True):
    """Idol weights + DINOv3 in models/idol, FLUX.2 VAE in models/vae."""
    missing = [n for n, size in IDOL_FILES.items()
               if not (os.path.exists(os.path.join(IDOL_DIR, n))
                       and os.path.getsize(os.path.join(IDOL_DIR, n)) == size)]
    if missing and not download:
        raise RuntimeError(
            "[idol] missing weights in models/idol: " + ", ".join(missing)
            + " - enable download_if_missing to fetch them from Hugging Face (speridlabs/idol).")
    for name in missing:
        _download(f"https://huggingface.co/speridlabs/idol/resolve/main/{name}",
                  os.path.join(IDOL_DIR, name), IDOL_FILES[name])

    dino_path = os.path.join(IDOL_DIR, DINO_NAME)
    if not (os.path.exists(dino_path) and os.path.getsize(dino_path) == DINO_BYTES):
        if not download:
            raise RuntimeError(f"[idol] missing {DINO_NAME} in models/idol (DINOv3 ViT-H/16+ encoder).")
        _download(DINO_URL, dino_path, DINO_BYTES)

    if not os.path.exists(VAE_LOCAL):
        if not download:
            raise RuntimeError(
                f"[idol] missing {VAE_LOCAL} - the FLUX.2 VAE is needed as an image encoder.")
        # the Comfy-Org copy is a few bytes different from some local repacks, so no size assert here;
        # the key remap + strict load below is the real validation.
        _download(VAE_URL, VAE_LOCAL)
    return dino_path


# --------------------------------------------------------------------------- encoders

def _remap_vae(state):
    """ComfyUI VAE keys -> diffusers AutoencoderKLFlux2 keys (same 251 tensors)."""
    def one(k):
        if k.startswith("encoder.quant_conv."):
            return k[len("encoder."):]
        if k.startswith("decoder.post_quant_conv."):
            return k[len("decoder."):]
        for side in ("encoder", "decoder"):
            if not k.startswith(side + "."):
                continue
            rest = k[len(side) + 1:]
            if rest.startswith("conv_in") or rest.startswith("conv_out") or k.startswith("bn."):
                return k
            if rest.startswith("norm_out."):
                return f"{side}.conv_norm_out." + rest.rsplit(".", 1)[-1]
            m = re.match(r"(down|up)\.(\d+)\.block\.(\d+)\.(.+)$", rest)
            if m:
                kind, i, j, tail = m.groups()
                blk = "down_blocks" if kind == "down" else "up_blocks"
                idx = int(i) if kind == "down" else (3 - int(i))  # diffusers runs up_blocks low-res first
                return f"{side}.{blk}.{idx}.resnets.{j}.{tail}".replace(".nin_shortcut.", ".conv_shortcut.")
            m = re.match(r"(down|up)\.(\d+)\.(downsample|upsample)\.conv\.(.+)$", rest)
            if m:
                kind, i, _, tail = m.groups()
                blk = "down_blocks" if kind == "down" else "up_blocks"
                idx = int(i) if kind == "down" else (3 - int(i))
                kind2 = "downsamplers" if kind == "down" else "upsamplers"
                return f"{side}.{blk}.{idx}.{kind2}.0.conv.{tail}"
            m = re.match(r"mid\.block_(\d)\.(.+)$", rest)
            if m:
                return f"{side}.mid_block.resnets.{int(m.group(1)) - 1}.{m.group(2)}"
            m = re.match(r"mid\.attn_1\.(.+)$", rest)
            if m:
                head, _, tail = m.group(1).partition(".")
                head = {"q": "to_q", "k": "to_k", "v": "to_v", "proj_out": "to_out", "norm": "group_norm"}.get(head, head)
                if head == "to_out":
                    tail = f"0.{tail}" if tail else "0"
                return f"{side}.mid_block.attentions.0.{head}" + (f".{tail}" if tail else "")
        return k

    out = {}
    for k, v in state.items():
        nk = one(k)
        # only the mid-block attention is stored as 1x1 convolutions upstream -> squeezed to Linear.
        if '.attentions.' in nk and v.dim() == 4 and v.shape[-1] == 1 and v.shape[-2] == 1:
            v = v.squeeze(-1).squeeze(-1)
        out[nk] = v
    return out


class IdolEncoders:
    """Drop-in replacement for idol.conditioning.ImageEncoder that avoids the two gated HF repos."""

    def __init__(self, dinov3_repo=None, flux2_vae_repo=None, dinov3_dtype="bf16", device="cuda"):
        import timm
        from diffusers.models import AutoencoderKLFlux2

        self.device = torch.device(device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dinov3_dtype]

        t0 = time.time()
        net = timm.create_model("vit_huge_plus_patch16_dinov3.lvd1689m", pretrained=False,
                               num_classes=0, img_size=1024)
        net.load_state_dict(load_file(os.path.join(IDOL_DIR, DINO_NAME)), strict=True)
        self.dinov3 = net.eval().requires_grad_(False).to(self.device, dtype)
        print(f"[idol] DINOv3 ViT-H/16+ ready in {time.time() - t0:.1f}s ({dtype})")

        t0 = time.time()
        vae = AutoencoderKLFlux2()
        vae.load_state_dict(_remap_vae(load_file(VAE_LOCAL)), strict=True)
        self.flux2_vae = vae.eval().requires_grad_(False).to(self.device, torch.bfloat16)
        print(f"[idol] FLUX.2 VAE ready in {time.time() - t0:.1f}s")

    def _dinov3_tokens(self, images):
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images.float() - mean) / std
        dtype = next(self.dinov3.parameters()).dtype
        tokens = self.dinov3.forward_features(images.to(dtype)).float()
        return F.layer_norm(tokens, tokens.shape[-1:])

    def _flux2_tokens(self, images):
        vae = self.flux2_vae
        latents = vae.encode(images.to(torch.bfloat16) * 2 - 1).latent_dist.sample()
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch, channels * 4, height // 2, width // 2)
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(latents.device, latents.dtype)
        tokens = ((latents - mean) / std).float().flatten(2).transpose(1, 2).contiguous()
        padding = torch.zeros(batch, 5, tokens.shape[2], dtype=tokens.dtype, device=tokens.device)
        return torch.cat([padding, tokens], dim=1)

    @torch.no_grad()
    def __call__(self, images):
        from tensordict import TensorDict
        images = images.to(self.device)
        feature1 = self._dinov3_tokens(images)
        feature2 = self._flux2_tokens(images)
        return TensorDict({"feature1": feature1, "feature2": feature2}, batch_size=feature1.shape[0])


idol_pipeline.ImageEncoder = IdolEncoders  # idol builds this internally; give it our local encoders


# --------------------------------------------------------------------------- helpers

def _get_pipeline(weights_dir, dinov3_dtype, download):
    key = (weights_dir, dinov3_dtype)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if key in _PIPE_CACHE:
        pipe = _PIPE_CACHE[key]
        _move_pipeline(pipe, device_type)  # no-op unless it was offloaded
        return pipe
    ensure_weights(download)
    if mm is not None:
        try:
            mm.free_memory(VRAM_REQUEST, mm.get_torch_device())
        except Exception:
            pass
    t0 = time.time()
    pipe = IdolPipeline(weights_dir, device=device_type, dinov3_dtype=dinov3_dtype)
    pipe._idol_device = device_type
    print(f"[idol] pipeline loaded in {time.time() - t0:.1f}s")
    _PIPE_CACHE[key] = pipe
    return pipe


def _pipe_modules(pipe):
    for attr in ("denoiser", "decoder", "decoder_gs", "background_remover"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            yield mod
    enc = getattr(pipe, "image_encoder", None)
    if enc is not None:
        for attr in ("dinov3", "flux2_vae"):
            mod = getattr(enc, attr, None)
            if mod is not None:
                yield mod


def _move_pipeline(pipe, device):
    """Move the pipeline (incl. both encoders) between cuda and cpu - keeps references alive, frees VRAM."""
    if getattr(pipe, "_idol_device", None) == device:
        return
    for mod in _pipe_modules(pipe):
        mod.to(device)
    pipe._idol_device = device
    if device == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def drop_pipeline(pipe=None):
    """Offload to CPU so the 3.7 GB leaves VRAM even while the graph still holds a reference.

    The pipeline object stays in the cache: the next run only pays the PCIe move back to the GPU
    (~2 s) instead of a 36 s rebuild from disk.
    """
    targets = [pipe] if pipe is not None else list(_PIPE_CACHE.values())
    for p in targets:
        try:
            _move_pipeline(p, "cpu")
        except Exception as e:
            print(f"[idol] offload failed: {e}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[idol] pipeline offloaded from VRAM")


def _to_pil(image_tensor, mask_tensor=None):
    arr = (image_tensor.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    pil = Image.fromarray(arr).convert("RGB")
    if mask_tensor is not None:
        a = (mask_tensor.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
        pil = pil.convert("RGBA")
        pil.putalpha(Image.fromarray(a))
    return pil


def _to_image_batch(pils):
    arrs = [np.asarray(p.convert("RGB"), dtype=np.float32) / 255.0 for p in pils]
    return torch.from_numpy(np.stack(arrs, axis=0))


def _splat_from_gaussian(g):
    """idol Gaussian (z-up, object frame) -> comfy Types.SPLAT (y-up, render-ready)."""
    xyz = g.xyz.detach().float()
    q = g.rotation.detach().float()
    # z-up -> y-up: (x, y, z) -> (x, -z, y) = rotate +90 deg about x, applied to positions and rotations
    pos = torch.stack([xyz[:, 0], -xyz[:, 2], xyz[:, 1]], dim=1)
    h = 0.5 ** 0.5
    rw = h * q[:, 0] - h * q[:, 1]
    rx = h * q[:, 1] + h * q[:, 0]
    ry = h * q[:, 2] - h * q[:, 3]
    rz = h * q[:, 3] + h * q[:, 2]
    rot = torch.stack([rw, rx, ry, rz], dim=1)
    rot = rot / rot.norm(dim=-1, keepdim=True)

    out_dev = mm.intermediate_device() if mm is not None else torch.device("cpu")
    return SPLAT_TYPE(
        pos.unsqueeze(0).to(out_dev),
        g.scaling.detach().float().unsqueeze(0).to(out_dev),
        rot.unsqueeze(0).to(out_dev),
        g.opacity.detach().float().unsqueeze(0).to(out_dev),
        g.features_dc.detach().float().unsqueeze(0).to(out_dev),  # (B,N,1,3) SH degree 0
    )


def _generate_prepared(pipe, prepared, seed, steps, cfg_strength, rescale_t, num_gaussians):
    from tensordict import TensorDict
    pixels = torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div(255.0)[None].to(pipe.device)
    torch.manual_seed(int(seed))
    cond = pipe.image_encoder(pixels)
    noise = TensorDict({
        "latent": torch.randn(1, pipe.denoiser.q_token_length, pipe.denoiser.in_channels, device=pipe.device),
        "camera": torch.randn(1, 1, pipe.denoiser.cam_channels, device=pipe.device),
    }, batch_size=1, device=pipe.device)
    sample = sample_flow(pipe.denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                         steps=int(steps), rescale_t=float(rescale_t), cfg_strength=float(cfg_strength),
                         verbose=False)
    gaussian = pipe.decode(sample["latent"], int(num_gaussians))
    return gaussian, decode_camera(sample["camera"][0, 0])


# --------------------------------------------------------------------------- nodes

class IdolLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "dinov3_dtype": (["bf16", "fp32"], {"default": "bf16",
                                                "tooltip": "fp32 runs the DINOv3 encoder in full precision (+1.7 GB VRAM)."}),
            "download_if_missing": ("BOOLEAN", {"default": True,
                                                "tooltip": "Fetch the 3.1 GB of idol weights and the DINOv3 encoder on first use."}),
            "weights_dir": ("STRING", {"default": "", "tooltip": "Optional override; empty = models/idol"}),
        }}

    RETURN_TYPES = ("IDOL_PIPELINE",)
    RETURN_NAMES = ("idol",)
    FUNCTION = "load"
    CATEGORY = "3d/idol"
    DESCRIPTION = "Load the Idol image-to-splat pipeline (1.8B, ~3.7 GB VRAM resident, 7.2 GB peak)."

    def load(self, dinov3_dtype, download_if_missing, weights_dir):
        wdir = weights_dir.strip() or IDOL_DIR
        pipe = _get_pipeline(wdir, dinov3_dtype, download_if_missing)
        return (pipe,)


class IdolPreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "idol": ("IDOL_PIPELINE",),
            "image": ("IMAGE",),
            "mode": (["synthetic", "real"], {"default": "synthetic",
                                             "tooltip": "synthetic: renders/generated objects (tight crop). real: photographs (looser crop)."}),
            "crop_margin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                      "tooltip": "0 = use the mode's default (1.2 synthetic / 1.7 real)."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Object alpha (white = keep). Skips BiRefNet background removal."}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preprocessed",)
    FUNCTION = "run"
    CATEGORY = "3d/idol"

    def run(self, idol, image, mode, crop_margin, mask=None):
        outs = []
        for i in range(image.shape[0]):
            m = mask[i] if (mask is not None and mask.shape[0] > min(i, mask.shape[0] - 1)) else None
            pil = _to_pil(image[i], m)
            margin = crop_margin if crop_margin > 0 else None
            outs.append(idol.preprocess(pil, mode=mode, crop_margin=margin))
        return (_to_image_batch(outs),)


class IdolGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "idol": ("IDOL_PIPELINE",),
            "image": ("IMAGE",),
            "mode": (["synthetic", "real"], {"default": "synthetic"}),
            "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1,
                              "tooltip": "Sampling steps. CFG doubles the cost: 50 steps = 100 DiT passes."}),
            "cfg_strength": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            "rescale_t": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 6.0, "step": 0.1,
                                    "tooltip": "Time shift: higher spends more steps on the overall shape."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            "num_gaussians": (GAUSSIAN_COUNTS, {"default": "262144"}),
            "crop_margin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            "save_ply": ("BOOLEAN", {"default": True, "tooltip": "Write a standard 3DGS .ply to output/idol/."}),
            "unload_after_run": ("BOOLEAN", {"default": False,
                                             "tooltip": "Offload the pipeline from VRAM when this run finishes (frees 3.7 GB; the next run reloads it)."}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "Object alpha (white = keep). Skips BiRefNet background removal."}),
        }}

    RETURN_TYPES = ("SPLAT", "IMAGE", "STRING")
    RETURN_NAMES = ("splat", "preprocessed", "ply_path")
    FUNCTION = "run"
    CATEGORY = "3d/idol"
    DESCRIPTION = "One image -> 3D Gaussian splat. Feed the SPLAT into RenderSplat (turntable) or SplatToFile3D."

    @torch.no_grad()
    def run(self, idol, image, mode, steps, cfg_strength, rescale_t, seed, num_gaussians, crop_margin,
            save_ply, unload_after_run=False, mask=None):
        n_gauss = int(num_gaussians)
        batch = image.shape[0]
        bar = comfy_utils.ProgressBar(batch) if comfy_utils is not None else None

        plies, pils, splats, cams = [], [], [], []
        out_dir = os.path.join(folder_paths.get_output_directory(), "idol")
        if save_ply:
            os.makedirs(out_dir, exist_ok=True)

        for i in range(batch):
            m = mask[i] if (mask is not None and mask.shape[0] > 0) else None
            pil = _to_pil(image[min(i, image.shape[0] - 1)], m)
            margin = crop_margin if crop_margin > 0 else None
            prepared = idol.preprocess(pil, mode=mode, crop_margin=margin)
            t0 = time.time()
            gaussian, camera = _generate_prepared(idol, prepared, seed, steps, cfg_strength,
                                                  rescale_t, n_gauss)
            print(f"[idol] {n_gauss} gaussians in {time.time() - t0:.1f}s "
                  f"(steps={steps}, cfg={cfg_strength}, seed={seed})")
            pils.append(prepared)
            splats.append(_splat_from_gaussian(gaussian))
            cams.append(camera)
            if save_ply:
                name = f"idol_{int(time.time())}_{i}_s{seed}_{n_gauss}.ply"
                path = os.path.join(out_dir, name)
                gaussian.save_ply(path)
                plies.append(path)
            if bar is not None:
                bar.update(1)

        if len(splats) == 1:
            splat = splats[0]
        else:  # different images -> keep them as a batch of equal-length rows
            splat = SPLAT_TYPE(
                torch.cat([s.positions for s in splats], 0),
                torch.cat([s.scales for s in splats], 0),
                torch.cat([s.rotations for s in splats], 0),
                torch.cat([s.opacities for s in splats], 0),
                torch.cat([s.sh for s in splats], 0),
            )
        ply_out = "\n".join(plies) if plies else ""
        if not save_ply:
            ply_out = json.dumps(cams[0] if len(cams) == 1 else cams)
        if unload_after_run:
            drop_pipeline(idol)
        return (splat, _to_image_batch(pils), ply_out)


class IdolUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {"idol": ("IDOL_PIPELINE",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "3d/idol"
    OUTPUT_NODE = True

    def run(self, idol=None):
        drop_pipeline(idol)
        return ()


NODE_CLASS_MAPPINGS = {
    "IdolLoader": IdolLoader,
    "IdolPreprocess": IdolPreprocess,
    "IdolGenerate": IdolGenerate,
    "IdolUnload": IdolUnload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IdolLoader": "Idol Loader",
    "IdolPreprocess": "Idol Preprocess",
    "IdolGenerate": "Idol Image to Splat",
    "IdolUnload": "Idol Unload VRAM",
}
