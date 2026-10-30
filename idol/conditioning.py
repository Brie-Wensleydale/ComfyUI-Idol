"""Image conditioning: DINOv3 ViT-H+/16 patch tokens and FLUX.2 VAE latents of the same 1024 x 1024 image.

The denoiser reads both token sequences side by side: DINOv3 gives 5 + 64 x 64 tokens (class, 4 registers,
patches), the FLUX.2 VAE latent is patchified 2 x 2 to 64 x 64 tokens and padded with 5 zero tokens so the two
sequences line up."""
import torch
import torch.nn.functional as F
from tensordict import TensorDict

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DTYPES = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}


class ImageEncoder:
    def __init__(self, dinov3_repo: str, flux2_vae_repo: str,
                 dinov3_dtype: str = 'bf16', device: str = 'cuda'):
        from diffusers.models import AutoencoderKLFlux2
        from transformers import AutoModel

        self.device = torch.device(device)
        self.dinov3 = AutoModel.from_pretrained(dinov3_repo, torch_dtype=DTYPES[dinov3_dtype])
        self.dinov3 = self.dinov3.eval().requires_grad_(False).to(self.device)
        self.flux2_vae = AutoencoderKLFlux2.from_pretrained(flux2_vae_repo, subfolder='vae',
                                                            torch_dtype=torch.bfloat16)
        self.flux2_vae = self.flux2_vae.eval().requires_grad_(False).to(self.device)

    def _dinov3_tokens(self, images):
        mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images.float() - mean) / std
        dtype = next(self.dinov3.parameters()).dtype
        tokens = self.dinov3(pixel_values=images.to(dtype)).last_hidden_state.float()
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
    def __call__(self, images: torch.Tensor) -> TensorDict:
        """images: [B, 3, 1024, 1024] in [0, 1] -> TensorDict(feature1=DINOv3 tokens, feature2=FLUX.2 VAE tokens)."""
        images = images.to(self.device)
        feature1 = self._dinov3_tokens(images)
        feature2 = self._flux2_tokens(images)
        return TensorDict({'feature1': feature1, 'feature2': feature2}, batch_size=feature1.shape[0])
