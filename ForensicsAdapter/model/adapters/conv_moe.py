import torch
from torch import nn
from torch.nn import functional as F


class _DifferenceConv2d(nn.Conv2d):
    """Convert a learned 3x3 kernel using the MoE-FFD difference operators."""

    def __init__(self, channels, operator):
        radius = 2 if operator == "rdc" else 1
        super().__init__(channels, channels, kernel_size=3, padding=radius)
        self.operator = operator
        self.effective_size = 2 * radius + 1

    def forward(self, features):
        weights = self.weight.flatten(2)
        if self.operator == "cdc":
            # Each surrounding coefficient multiplies (neighbor - center).
            kernel = weights.clone()
            kernel[..., 4] = 2 * weights[..., 4] - weights.sum(dim=-1)
        elif self.operator == "adc":
            # Pair each surrounding coefficient with the next location on the ring.
            clockwise = (3, 0, 1, 6, 4, 2, 7, 8, 5)
            kernel = weights - weights[..., clockwise]
            kernel[..., 4] = weights[..., 4]
        else:
            # Each radial coefficient pairs positions at distance one and two.
            inner_ring = (6, 7, 8, 11, 13, 16, 17, 18)
            outer_ring = (0, 2, 4, 10, 14, 20, 22, 24)
            kernel = weights.new_zeros((*weights.shape[:-1], 25))
            kernel[..., 12] = weights[..., 0]
            kernel[..., inner_ring] = -weights[..., 1:]
            kernel[..., outer_ring] = weights[..., 1:]
        kernel = kernel.reshape(*weights.shape[:-1], self.effective_size, self.effective_size)
        return F.conv2d(features, kernel, self.bias, padding=self.padding)


class ConvMoE(nn.Module):
    """Build a visual-token residual from shared convolution and routed experts."""

    expert_names = ("cdc", "adc", "rdc")

    def __init__(self, channels, hidden_dim=16, top_k=1):
        super().__init__()
        if type(top_k) is not int or top_k not in (1, 2):
            raise ValueError(f"Invalid ConvMoE top_k: {top_k!r}. Expected integer 1 or 2.")
        self.top_k = top_k
        self.down = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.shared_expert = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.experts = nn.ModuleList(
            _DifferenceConv2d(hidden_dim, operator) for operator in self.expert_names
        )
        self.router = nn.Linear(2 * hidden_dim, len(self.experts))
        self.up = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, spatial_tokens, spatial_shape):
        height, width = spatial_shape
        batch, patch_count, channels = spatial_tokens.shape
        if min(batch, height, width) < 1 or patch_count != height * width:
            raise ValueError(f"Visual token shape {spatial_tokens.shape} does not match grid {spatial_shape}")

        grid = spatial_tokens.transpose(1, 2).reshape(batch, channels, height, width)
        features = self.down(grid)
        with torch.autocast(device_type=features.device.type, enabled=False):
            values = features.float()
            local_mean = F.avg_pool2d(values, kernel_size=3, stride=1, padding=1, count_include_pad=False)
            global_mean = values.mean(dim=(2, 3))
            local_deviation = (values - local_mean).abs().mean(dim=(2, 3))
            descriptor = torch.cat((global_mean, local_deviation), dim=1)
            logits = F.linear(descriptor, self.router.weight.float(), self.router.bias.float())
            probabilities = logits.softmax(dim=1)
            importance = probabilities.mean(dim=0)
            balance_loss = (len(self.experts) * importance - 1).square().mean()
            selected_logits, selected = logits.topk(self.top_k, dim=1)
            if self.top_k == 1:
                # The full-softmax score keeps the top-1 weight differentiable.
                selected_weights = probabilities.gather(1, selected)
            else:
                selected_weights = selected_logits.softmax(dim=1)
            routing_weights = torch.zeros_like(logits).scatter(1, selected, selected_weights)

        mixed = self.shared_expert(features)
        for index, expert in enumerate(self.experts):
            weights = routing_weights[:, index, None, None, None].to(mixed.dtype)
            mixed = mixed + weights * expert(features)
        delta = self.up(F.gelu(mixed)).flatten(2).transpose(1, 2)
        return delta.to(spatial_tokens.dtype), balance_loss, routing_weights.detach()
