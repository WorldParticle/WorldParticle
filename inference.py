import os
import torch
import numpy as np
import argparse
import yaml
import mdtraj as md
from trajectory_model import TrajectoryModel
from TrajDataset import (
    load_fluid_sample_by_data,
    load_trajectory_sample_by_data,
    load_npz_sample_by_data,
    is_obstacle_npz_data,
    is_no_obstacle_npz_data,
    uses_obstacle_features,
    uses_force_position_override,
)
from torch.cuda.amp import autocast

@torch.no_grad()
def inference(args, cfg):
    device = cfg["device"] if torch.cuda.is_available() else "cpu"

    topology_neighbors_index = None
    topology_neighbors_row_splits = None
    force_traj = None
    feats = None
    box = None
    box_feats = None

    # === Load npz data ===
    if cfg["data"] == "trajectory":
        input_npz = os.path.join(args.data_dir, args.sample, args.sample + ".npz")
        pdb_file = os.path.join(args.data_dir, args.sample, args.sample + ".pdb")
        sample = load_trajectory_sample_by_data(
            npz_path=input_npz,
            pdb_path=pdb_file,
            frame_start=cfg.get("frame_start", 0),
            frame_end=cfg.get("frame_end", 100),
            return_topology=True,
        )
        positions = sample["positions"][args.start:args.end].cpu().numpy()
        velocities = sample["velocities"][args.start:args.end].cpu().numpy()
        feats = sample["feats"].to(device).half()
        topology = sample["topology"]
    
        
    elif is_no_obstacle_npz_data(cfg["data"]):
        npz_path = os.path.join(args.data_dir, args.sample, args.sample + ".npz")
        sample = load_npz_sample_by_data(
            data_type=cfg["data"],
            npz_path=npz_path,
            frame_start=cfg.get("frame_start", 0),
            frame_end=cfg.get("frame_end", 100),
        )
        positions = sample["positions"][args.start:args.end].cpu().numpy()
        velocities = sample["velocities"][args.start:args.end].cpu().numpy()
        feats = sample["feats"].to(device)
        topology_neighbors_index = (
            sample["topology_neighbors_index"].to(device)
            if sample["topology_neighbors_index"] is not None
            else None
        )
        topology_neighbors_row_splits = (
            sample["topology_neighbors_row_splits"].to(device)
            if sample["topology_neighbors_row_splits"] is not None
            else None
        )
        force_traj = (
            sample["force_traj"][args.start:args.end].to(device)
            if sample["force_traj"] is not None
            else None
        )
    elif cfg["data"] == "fluid":
        data_path = os.path.join(args.data_dir, "train")
        sample = load_fluid_sample_by_data(
            data_path=data_path,
            sim_name=args.sample,
            start_idx=0,
            end_idx=15,
        )
        positions = sample["positions"][args.start:args.end].cpu().numpy()
        velocities = sample["velocities"][args.start:args.end].cpu().numpy()
        feats = sample["feats"].to(device)
        box = sample["box"].to(device)
        box_feats = sample["box_feats"].to(device)
    
    elif is_obstacle_npz_data(cfg["data"]):
        npz_path = os.path.join(args.data_dir, args.sample, args.sample + ".npz")
        sample = load_npz_sample_by_data(
            data_type=cfg["data"],
            npz_path=npz_path,
            frame_start=cfg.get("frame_start", 0),
            frame_end=cfg.get("frame_end", 60),
        )
        positions = sample["positions"][args.start:args.end].cpu().numpy()
        velocities = sample["velocities"][args.start:args.end].cpu().numpy()
        feats = sample["feats"].to(device)
        box = sample["box"].to(device)
        box_feats = sample["box_feats"].to(device)
        topology_neighbors_index = (
            sample["topology_neighbors_index"].to(device)
            if sample["topology_neighbors_index"] is not None
            else None
        )
        topology_neighbors_row_splits = (
            sample["topology_neighbors_row_splits"].to(device)
            if sample["topology_neighbors_row_splits"] is not None
            else None
        )
        force_traj = (
            sample["force_traj"][args.start:args.end].to(device)
            if sample["force_traj"] is not None
            else None
        )
    else:
        raise ValueError(f"Unsupported data type for inference: {cfg['data']}")

    # Resolve feature channels directly from loaded tensors.
    if feats is None:
        raise ValueError("Failed to load feats for inference.")
    cfg["resolved_other_feats_channels"] = int(feats.shape[-1]) if feats.dim() > 1 else 1
    if box_feats is not None:
        cfg["resolved_obstacle_feats_channels"] = int(box_feats.shape[-1]) if box_feats.dim() > 1 else 1
    else:
        cfg["resolved_obstacle_feats_channels"] = 0

    # Keep model-side config aligned with inference data root.
    cfg["data_folder"] = args.data_dir

    # === Construct model ===
    model = TrajectoryModel(cfg).to(device)

    # === Load checkpoint ===
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_state = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.model.load_state_dict(model_state, strict=False)
    model.eval()
    print(f"✅ Loaded checkpoint: {args.checkpoint}")

    # === initial state ===
    pos_i = torch.tensor(positions[0], dtype=torch.float32, device=device)
    vel_i = torch.tensor(velocities[0], dtype=torch.float32, device=device)

    pred_positions = [pos_i.squeeze(0).cpu().numpy()]
    pred_velocities = [vel_i.squeeze(0).cpu().numpy()]
    printed_force_override = False

    # === Autoregressive rollout ===
    num_steps = args.num_steps
    for step in range(num_steps):
        force_step = None
        if force_traj is not None and force_traj.numel() > 0:
            force_step = force_traj[min(step, force_traj.shape[0] - 1)]
        with autocast(enabled=False):
            with autocast(dtype=torch.bfloat16):

                if uses_obstacle_features(cfg["data"]):
                    pos_pred, vel_pred = model(
                        pos_i.float(),
                        vel_i.float(),
                        feats.float(),
                        box.float(),
                        box_feats.float(),
                        topology_neighbors_index=topology_neighbors_index,
                        topology_neighbors_row_splits=topology_neighbors_row_splits,
                        external_force=force_step.float() if force_step is not None else None,
                    )
                elif is_no_obstacle_npz_data(cfg["data"]):
                    pos_pred, vel_pred = model(
                        pos_i.float(),
                        vel_i.float(),
                        feats.float(),
                        None,
                        None,
                        topology_neighbors_index=topology_neighbors_index,
                        topology_neighbors_row_splits=topology_neighbors_row_splits,
                        external_force=force_step.float() if force_step is not None else None,
                    )
                elif cfg["data"] == "trajectory":
                    pos_pred, vel_pred = model(
                        pos_i.float(),
                        vel_i.float(),
                        feats.float(),
                        None,
                        None,
                        topology_neighbors_index=topology_neighbors_index,
                        topology_neighbors_row_splits=topology_neighbors_row_splits,
                        external_force=force_step.float() if force_step is not None else None,
                    )

        if (
            uses_force_position_override(cfg["data"])
            and force_step is not None
            and (step + 1) < len(positions)
        ):
            force_mask = torch.any(force_step != 0, dim=-1, keepdim=True)
            if torch.any(force_mask):
                if not printed_force_override:
                    print(
                        "[Diag] force position override enabled: particles with non-zero current-frame force use GT positions."
                    )
                    printed_force_override = True
                gt_pos_step = torch.tensor(positions[step + 1], dtype=pos_pred.dtype, device=device)
                pos_pred = torch.where(force_mask, gt_pos_step, pos_pred)

        pos_i, vel_i = pos_pred, vel_pred
        pred_positions.append(pos_pred.squeeze(0).cpu().numpy())
        pred_velocities.append(vel_pred.squeeze(0).cpu().numpy())

    pred_positions = np.stack(pred_positions, axis=0)
    pred_velocities = np.stack(pred_velocities, axis=0)

    # === Save results ===
    npz_path = os.path.join(args.output_dir, "outcomes", "npz", args.sample, f"{args.sample}_{args.start}_{args.end}.npz")

    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    
    np.savez(
        npz_path,
        position=pred_positions,
        velocity=pred_velocities,
    )
    print(f"✅ Saved predicted trajectory to {npz_path}")

    if cfg["data"] == "trajectory":
        xtc_path = os.path.join(args.output_dir, "outcomes", "xtc", args.sample, f"{args.sample}_pred_{args.dt}.xtc")
        os.makedirs(os.path.dirname(xtc_path), exist_ok=True)
        traj = md.Trajectory(xyz=pred_positions, topology=topology)
        traj.save_xtc(xtc_path)

    # === Compare with ground truth ===
    if args.dt < 1:
        raise ValueError(f"--dt must be >= 1, got {args.dt}.")

    gt_positions_sampled = positions[::args.dt]
    gt_velocities_sampled = velocities[::args.dt] if velocities is not None else None
    pred_positions_sampled = pred_positions[::args.dt]
    pred_velocities_sampled = pred_velocities[::args.dt]
    max_steps = min(len(gt_positions_sampled), len(pred_positions_sampled))
    if max_steps > 0:
        gt_positions = gt_positions_sampled[:max_steps]
        gt_velocities = gt_velocities_sampled[:max_steps] if gt_velocities_sampled is not None else None
        pred_positions_trimmed = pred_positions_sampled[:max_steps]
        pred_velocities_trimmed = pred_velocities_sampled[:max_steps]

        mse_pos = np.mean((gt_positions - pred_positions_trimmed) ** 2)
        mse_vel = np.mean((gt_velocities - pred_velocities_trimmed) ** 2) if gt_velocities is not None else None

        if mse_vel is not None:
            print(f"Position MSE: {mse_pos:.8f}, Velocity MSE: {mse_vel:.8f}")
        else:
            print(f"Position MSE: {mse_pos:.8f}")

    else:
        print("⚠️ Not enough ground truth frames for comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to sample trajectory folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save output npz and xtc")
    parser.add_argument("--sample", type=str, required=True, help="Sample for inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--dt", type=int, default=1, help="Sampling stride used for ground-truth comparison")
    parser.add_argument("--num_steps", type=int, default=300, help="Number of rollout steps to predict")
    parser.add_argument("--start", type=int, default=0, help="Number of rollout steps to predict")
    parser.add_argument("--end", type=int, default=300, help="Number of rollout steps to predict")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    inference(args, cfg)
