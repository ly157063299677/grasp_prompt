from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_prompt.models.attention import PromptDecoderBlock, TemporalEncoderBlock
from grasp_prompt.utils import mlp, sinusoidal_embedding


class Denoiser(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        ablation_cfg = cfg.get("ablation", {})
        dim = int(model_cfg["hidden_dim"])
        heads = int(model_cfg["num_heads"])
        max_prompts = int(cfg["data"]["max_prompts"])
        self.point_embed = mlp(3, dim, dim, layers=2)
        self.time_mlp = mlp(dim, dim * 2, dim, layers=2)
        self.encoders = nn.ModuleList(
            [
                TemporalEncoderBlock(
                    dim=dim,
                    num_heads=heads,
                    max_tokens=max_prompts,
                    local_k=int(model_cfg.get("tda_local_k", 4)),
                    use_tda=bool(ablation_cfg.get("use_tda", True)),
                )
                for _ in range(int(model_cfg["num_encoder_layers"]))
            ]
        )
        self.decoders = nn.ModuleList(
            [
                PromptDecoderBlock(
                    dim=dim,
                    num_heads=heads,
                    use_cdci=bool(ablation_cfg.get("use_cdci", True)),
                )
                for _ in range(int(model_cfg["num_decoder_layers"]))
            ]
        )
        self.out = mlp(dim, dim, 3, layers=2)

    def forward(
        self,
        noisy_points: torch.Tensor,
        timesteps: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> torch.Tensor:
        dim = condition_tokens.shape[-1]
        time_emb = self.time_mlp(sinusoidal_embedding(timesteps, dim))
        x = self.point_embed(noisy_points) + time_emb.unsqueeze(1)
        for block in self.encoders:
            x = block(x, noisy_points, time_emb)
        for block in self.decoders:
            x = block(x, condition_tokens)
        return self.out(x)


class GaussianDiffusion(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        steps = int(model_cfg["diffusion_steps"])
        betas = torch.linspace(float(model_cfg["beta_start"]), float(model_cfg["beta_end"]), steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        posterior_var = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)

        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("posterior_var", posterior_var.clamp_min(1e-20))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ab = self._extract(self.sqrt_alpha_bar, timesteps, x0)
        sqrt_om = self._extract(self.sqrt_one_minus_alpha_bar, timesteps, x0)
        return sqrt_ab * x0 + sqrt_om * noise

    def training_loss(
        self,
        denoiser: Denoiser,
        x0: torch.Tensor,
        condition_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x0.shape[0]
        timesteps = torch.randint(0, self.steps, (bsz,), device=x0.device)
        noise = torch.randn_like(x0)
        noisy = self.q_sample(x0, timesteps, noise)
        pred = denoiser(noisy, timesteps, condition_tokens)
        loss = F.mse_loss(pred, noise)
        return loss, pred

    @torch.no_grad()
    def sample(
        self,
        denoiser: Denoiser,
        condition_tokens: torch.Tensor,
        shape: tuple[int, int, int],
        steps: int | None = None,
    ) -> torch.Tensor:
        device = condition_tokens.device
        x = torch.randn(shape, device=device)
        max_step = self.steps if steps is None else min(int(steps), self.steps)
        step_ids = torch.linspace(self.steps - 1, 0, max_step, device=device).long()
        for step in step_ids:
            t = torch.full((shape[0],), int(step.item()), device=device, dtype=torch.long)
            eps = denoiser(x, t, condition_tokens)
            beta_t = self._extract(self.betas, t, x)
            alpha_t = self._extract(self.alphas, t, x)
            alpha_bar_t = self._extract(self.alpha_bar, t, x)
            mean = (x - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_t)
            if int(step.item()) > 0:
                var = self._extract(self.posterior_var, t, x)
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, timesteps)
        while out.ndim < target.ndim:
            out = out.unsqueeze(-1)
        return out
