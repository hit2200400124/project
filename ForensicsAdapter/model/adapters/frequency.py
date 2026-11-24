import math

import torch
from torch import nn
from torch.nn import functional as F


class FrequencyFusion(nn.Module):
    """Add aligned DCT features to the visual tokens after Adapter block 3."""

    patch_size = 16
    modes = ("absolute", "uniform", "guided")

    def __init__(self, token_dim, mode):
        super().__init__()
        if mode not in self.modes:
            raise ValueError(f"Invalid frequency_fusion mode: {mode}")
        self.mode = mode
        self.coverage_head = nn.Linear(token_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, token_dim),
        )
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

        size = self.patch_size
        indices = torch.arange(size, dtype=torch.float32)
        basis = torch.cos((math.pi / size) * indices[:, None] * (indices + 0.5))
        basis *= math.sqrt(2.0 / size)
        basis[0] = 1.0 / math.sqrt(size)
        self.register_buffer("dct_basis", basis, persistent=False)

        diagonal = indices[:, None] + indices
        bands = torch.stack(((diagonal >= 4) & (diagonal < 12), diagonal >= 12), dim=-1)
        weights = bands.reshape(size * size, 2).float()
        self.register_buffer("band_weights", weights / weights.sum(dim=0), persistent=False)

    def band_energy(self, luma):
        size = self.patch_size
        with torch.autocast(device_type=luma.device.type, enabled=False):
            # The patch axes follow the row-major order of the Adapter's patch convolution.
            patches = luma.float().unfold(2, size, size).unfold(3, size, size).squeeze(1)
            basis = self.dct_basis.float()
            coefficients = basis @ patches @ basis.T
            energy = coefficients.square().flatten(-2) @ self.band_weights.float()
            return (energy + 1e-8).log()

    @staticmethod
    def neighbor_sum(grid):
        # Sum the in-image neighbors in a 5x5 window, excluding the center.
        return F.avg_pool2d(grid, kernel_size=5, stride=1, padding=2, divisor_override=1) - grid

    def relative_energy(self, energy, coverage_logits):
        batch, height, width, _ = energy.shape
        if self.mode == "absolute":
            used = torch.zeros((batch, height * width), device=energy.device, dtype=torch.bool)
            return torch.zeros_like(energy), used

        coverage = coverage_logits.detach().float().sigmoid().reshape(batch, 1, height, width)
        candidates = (coverage <= 0.2).to(energy.dtype)
        candidate_count = self.neighbor_sum(candidates)
        use_reference = candidate_count >= 3
        weights = candidates if self.mode == "guided" else torch.ones_like(candidates)
        reference_count = candidate_count if self.mode == "guided" else self.neighbor_sum(weights)

        energy_grid = energy.permute(0, 3, 1, 2)
        reference = self.neighbor_sum(energy_grid * weights) / reference_count.clamp_min(1)
        delta = (energy_grid - reference).masked_fill(~use_reference, 0)
        return delta.permute(0, 2, 3, 1), use_reference.flatten(1)

    def forward(self, spatial_tokens, luma, spatial_shape):
        height, width = spatial_shape
        expected_shape = (spatial_tokens.shape[0], 1, height * self.patch_size, width * self.patch_size)
        if luma is None or luma.shape != expected_shape:
            raise ValueError(f"Frequency fusion requires pre-normalization 'luma' of shape {expected_shape}")

        coverage_logits = self.coverage_head(spatial_tokens.detach()).squeeze(-1)
        energy = self.band_energy(luma)
        delta, reference_used = self.relative_energy(energy, coverage_logits)
        features = torch.cat((energy, delta), dim=-1).flatten(1, 2)
        fused = spatial_tokens + self.projection(features.to(spatial_tokens.dtype))
        return fused, coverage_logits, reference_used

    @classmethod
    def coverage_loss(cls, logits, labels, masks, mask_available):
        real = labels == 0
        target = torch.zeros_like(logits, dtype=torch.float32)
        valid = real
        if masks is not None:
            targets = F.avg_pool2d(
                masks.movedim(-1, 1).gt(127).float(), kernel_size=cls.patch_size
            ).flatten(1)
            target = torch.where(real[:, None], target, targets)
            valid = torch.ones_like(real) if mask_available is None else real | mask_available

        loss = F.binary_cross_entropy_with_logits(logits.float(), target, reduction="none").mean(dim=1)
        return (loss * valid).sum() / valid.sum().clamp_min(1)
