"""Particle model wrapper for checkpoint-based inference."""

import torch
from pytorch_lightning import LightningModule

from models.particle_network_cross_attn_feat import ParticleNetworkCrossAttnLocalFeature
from TrajDataset import (
    uses_obstacle_features,
    uses_topology_neighbors,
    get_default_topology_search_radius,
)


def _resolve_topology_settings(cfg):
    """Resolve topology conv switch and radius from built-in data rules + manual overrides."""
    data_type = cfg.get("data", None)

    use_topology_conv = cfg.get("use_topology_conv", None)
    if use_topology_conv is None:
        resolved_use_topology_conv = bool(uses_topology_neighbors(data_type))
    else:
        resolved_use_topology_conv = bool(use_topology_conv)

    topology_search_radius = cfg.get("topology_search_radius", None)
    if topology_search_radius is None and resolved_use_topology_conv:
        topology_search_radius = get_default_topology_search_radius(data_type)

    if topology_search_radius is not None:
        topology_search_radius = float(topology_search_radius)

    return resolved_use_topology_conv, topology_search_radius

class TrajectoryModel(LightningModule):
    """Construct the particle network and preserve its checkpoint layout."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        resolved_use_topology_conv, resolved_topology_search_radius = _resolve_topology_settings(cfg)
        self.cfg["resolved_use_topology_conv"] = resolved_use_topology_conv
        self.cfg["resolved_topology_search_radius"] = resolved_topology_search_radius
        # The caller resolves feature dimensions from the input sample.
        self.cfg["resolved_other_feats_channels"] = int(cfg["resolved_other_feats_channels"])
        self.cfg["resolved_obstacle_feats_channels"] = int(cfg["resolved_obstacle_feats_channels"])

        # -------- Model --------
        self.model = ParticleNetworkCrossAttnLocalFeature(
            # ===== Continuous Convolution =====
            kernel_size=cfg.get("kernel_size", [4, 4, 4]),
            cconv_embedding_dim=cfg.get("cconv_embedding_dim", 32),
            radius_scale=cfg.get("radius_scale", 1.5),
            particle_radius=cfg.get("particle_radius", 0.025),
            data=cfg["data"],
            use_topology_conv=resolved_use_topology_conv,
            topology_search_radius=resolved_topology_search_radius,
            topology_blend_alpha=cfg.get("topology_blend_alpha", None),
            other_feats_channels=self.cfg["resolved_other_feats_channels"],
            obstacle_feats_channels=self.cfg["resolved_obstacle_feats_channels"],

            # ===== Slot / Super-particle =====
            slot_compression_ratio=cfg.get("slot_compression_ratio", 10),
            slot_attn_iters=cfg.get("slot_attn_iters", 3),
            slot_attn_heads=cfg.get("slot_attn_heads", 6),

            # ===== RoPE =====
            particle_position_rope_dim=cfg.get("particle_position_rope_dim", 96),

            # ===== Decoder =====
            num_decoder_layers=cfg.get("num_decoder_layers", 3),
            decoder_attn_heads=cfg.get("decoder_attn_heads", 4),
            decoder_attn_ffn_hidden_dim=cfg.get("decoder_attn_ffn_hidden_dim", 384),
            decoder_attn_dropout=cfg.get("decoder_attn_dropout", 0.1),

            # ===== Output MLP =====
            output_hidden_dim=cfg.get("output_hidden_dim", 128),
            output_layers=cfg.get("output_layers", 2),

            # ===== Physics =====
            gravity=cfg["gravity"],
            timestep=cfg.get("timestep", 0.02),

            # ===== Debug =====
            verbose=cfg.get("verbose", False),
            velhead = cfg.get("velhead", False),
        )

    @staticmethod
    def _flatten_topology_neighbors(topology_neighbors_index, topology_neighbors_row_splits, B, N, device):
        if topology_neighbors_index is None or topology_neighbors_row_splits is None:
            return None, None

        if topology_neighbors_index.dim() == 1:
            topology_neighbors_index = topology_neighbors_index.unsqueeze(0).expand(B, -1)
        if topology_neighbors_row_splits.dim() == 1:
            topology_neighbors_row_splits = topology_neighbors_row_splits.unsqueeze(0).expand(B, -1)

        idx_parts = []
        row_parts = [torch.zeros(1, dtype=torch.long, device=device)]
        edge_offset = 0

        for b in range(B):
            idx_b = topology_neighbors_index[b].to(device=device, dtype=torch.long).contiguous()
            row_b = topology_neighbors_row_splits[b].to(device=device, dtype=torch.long).contiguous()

            if row_b.numel() != N + 1:
                raise ValueError(
                    f"topology_neighbors_row_splits size mismatch in batch {b}: "
                    f"got {row_b.numel()}, expected {N + 1}"
                )

            idx_b = idx_b + b * N
            row_shifted = row_b[1:] + edge_offset
            edge_offset = int(row_shifted[-1].item())

            idx_parts.append(idx_b)
            row_parts.append(row_shifted)

        topo_idx_cat = torch.cat(idx_parts, dim=0)
        topo_row_cat = torch.cat(row_parts, dim=0)
        return topo_idx_cat, topo_row_cat

    def forward(self, pos, vel, feats, box, box_feats,
                topology_neighbors_index=None, topology_neighbors_row_splits=None,
                external_force=None):
        """
        Forward pass supporting batched input:
        - pos, vel: [B, N, 3]
        - feats: [B, N, F]
        - box, box_feats: [B, M, 3], [B, M, Fb]

        Returns:
            pos_corrected, vel_corrected: [B, N, 3]
        """
        if pos.ndim == 2:
            model_inputs = (
                (pos, vel, feats, box, box_feats)
                if uses_obstacle_features(self.cfg["data"])
                else (pos, vel, feats, None, None)
            )
            if topology_neighbors_index is not None and topology_neighbors_row_splits is not None:
                model_inputs = model_inputs + (topology_neighbors_index, topology_neighbors_row_splits)
            return self.model(
                model_inputs,
                k_partite=self.cfg.get("k_partite", -1),
                external_force=external_force,
            )

        if pos.ndim != 3:
            raise ValueError(f"Expected pos with shape [N, 3] or [B, N, 3], got {tuple(pos.shape)}")

        batch_size, num_particles, coord_dim = pos.shape
        if coord_dim != 3 or vel.shape != pos.shape:
            raise ValueError(
                f"Batched pos/vel must both have shape [B, N, 3], got {tuple(pos.shape)} and {tuple(vel.shape)}"
            )
        if feats.shape[:2] != (batch_size, num_particles):
            raise ValueError(
                f"Batched feats must start with [B, N]={batch_size, num_particles}, got {tuple(feats.shape)}"
            )

        pos_cat = pos.reshape(batch_size * num_particles, 3)
        vel_cat = vel.reshape(batch_size * num_particles, 3)
        feats_cat = feats.reshape(batch_size * num_particles, -1)
        force_cat = (
            external_force.reshape(batch_size * num_particles, 3)
            if external_force is not None
            else None
        )
        batch_splits = torch.arange(
            batch_size + 1, device=pos.device, dtype=torch.int64
        ) * num_particles
        batch_info = {
            "batch_size": batch_size,
            "batch_splits": batch_splits,
        }

        if uses_obstacle_features(self.cfg["data"]):
            if box is None or box_feats is None or box.ndim != 3 or box_feats.ndim != 3:
                raise ValueError("Batched obstacle data requires box and box_feats with shapes [B, M, C].")
            if box.shape[0] != batch_size or box_feats.shape[:2] != box.shape[:2]:
                raise ValueError(
                    f"Batched box/box_feats mismatch: {tuple(box.shape)} vs {tuple(box_feats.shape)}"
                )
            num_obstacles = box.shape[1]
            box_cat = box.reshape(batch_size * num_obstacles, -1)
            box_feats_cat = box_feats.reshape(batch_size * num_obstacles, -1)
            batch_info["box_splits"] = (
                torch.arange(batch_size + 1, device=pos.device, dtype=torch.int64)
                * num_obstacles
            )
        else:
            box_cat = None
            box_feats_cat = None

        topo_idx_cat, topo_row_cat = self._flatten_topology_neighbors(
            topology_neighbors_index,
            topology_neighbors_row_splits,
            batch_size,
            num_particles,
            pos.device,
        )
        model_inputs = (pos_cat, vel_cat, feats_cat, box_cat, box_feats_cat)
        if topo_idx_cat is not None and topo_row_cat is not None:
            model_inputs = model_inputs + (topo_idx_cat, topo_row_cat)

        pos_corr_cat, vel_corr_cat = self.model(
            model_inputs,
            batch_info=batch_info,
            k_partite=self.cfg.get("k_partite", -1),
            external_force=force_cat,
        )
        return (
            pos_corr_cat.reshape(batch_size, num_particles, 3),
            vel_corr_cat.reshape(batch_size, num_particles, 3),
        )
