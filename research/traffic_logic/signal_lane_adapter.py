"""Small signal residual for an existing DiT lane embedding.

The point mask selects source geometry; it does not define a stopping region.
This module does not change the original model or its checkpoint schema.
"""
import torch
from torch import nn


class SignalLaneAdapter(nn.Module):
    """Encode [nine state flags, stop XY, masked local mean XY] per signal/lane."""

    def __init__(self, output_dim, hidden_dim=64):
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Adapter dimensions must be positive")
        self.input = nn.Linear(13, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, point_mask, road_points, active):
        if features.ndim != 2 or features.shape[1] != 11:
            raise ValueError("Expected signal features [S,11]")
        if road_points.ndim != 3 or road_points.shape[-1] != 2:
            raise ValueError("Expected road geometry [L,P,2]")
        if tuple(point_mask.shape) != (features.shape[0], *road_points.shape[:2]):
            raise ValueError("Expected point mask [S,L,P]")
        if tuple(active.shape) != (features.shape[0],):
            raise ValueError("Expected active flags [S]")
        if point_mask.dtype != torch.bool or active.dtype != torch.bool:
            raise ValueError("Mask and active flags must be boolean")
        if not torch.isfinite(features).all() or not torch.isfinite(road_points).all():
            raise ValueError("Signal features and geometry must be finite")
        if not (features.device == point_mask.device == road_points.device == active.device):
            raise ValueError("All condition tensors must be on the same device")
        dtype = self.input.weight.dtype
        features, road_points = features.to(dtype), road_points.to(dtype)
        weights = point_mask.to(dtype)
        count = weights.sum(-1)
        local_xy = torch.einsum("slp,lpd->sld", weights, road_points) / count.clamp_min(1).unsqueeze(-1)
        tiled = features[:, None].expand(-1, road_points.shape[0], -1)
        encoded = self.output(torch.relu(self.input(torch.cat([tiled, local_xy], -1))))
        valid = point_mask.any(-1) & active[:, None]
        encoded = encoded * valid.unsqueeze(-1)
        # Each supported signal contributes once to its lane, regardless of point count.
        return encoded.sum(0) / valid.sum(0).clamp_min(1).unsqueeze(-1)


class LaneSignalResidual:
    """Scoped forward hook; the adapter is not registered inside the base model."""

    def __init__(self, lane_embedder, adapter, features, point_mask, road_points, active, enabled=True):
        self.lane_embedder = lane_embedder
        self.adapter = adapter
        self.condition = (features, point_mask, road_points, active)
        self.enabled = enabled
        self.calls = 0
        self.residual_l2 = 0.0
        self.handle = None

    def _inject(self, module, arguments, output):
        residual = self.adapter(*self.condition).to(output.dtype)
        if residual.shape != output.shape:
            raise ValueError("Signal residual shape differs from lane embedding")
        self.calls += 1
        self.residual_l2 = float(residual.detach().norm())
        return output + residual

    def __enter__(self):
        if self.handle is not None:
            raise RuntimeError("Signal hook context is already active")
        if self.enabled:
            self.handle = self.lane_embedder.register_forward_hook(self._inject)
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False
