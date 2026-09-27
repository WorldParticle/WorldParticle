import os
import numpy as np
import mdtraj as md
import torch
from torch.utils.data import Dataset
import random
import math
import random
from collections import defaultdict
from torch.utils.data import Sampler, BatchSampler, get_worker_info
import torch.distributed as dist
from tqdm import tqdm
import os
import zstandard as zstd
import msgpack
import numpy as np

standard_elements = [
    'C', 'H', 'O', 'N', 'S', 'P', 
]
element_to_idx = {elem: i for i, elem in enumerate(standard_elements)}
UNK_IDX = len(standard_elements)

OBSTACLE_NPZ_DATA = {
    "sand",
    "elastic",
    "cloth_on_ball",
    "cloth_on_needle",
    "elastic_drag",
    "elastic_drop",
    "non",
    "non_obj",
    "non_bunny",
    "fancy_fluid",
    "cloth_no_self_attention",
    "cloth_no_ffn",
    "cloth_no_all",
    "fancy_flulid_div",
    "fancy_fluid_div",
    "real_world",
    "real_world_override",
    "duck_drop",
    "hair_ball",
}

NO_OBSTACLE_NPZ_DATA = {
    "flag",
    "cloth",  # backward-compatible alias
    "fancy_fluid_abl",
    "local_fluid",
    "twist",
    "twist_force",
    "twist_force_override",
}

TOPOLOGY_ENABLED_DATA = {
    "elastic",
    "cloth_on_ball",
    "cloth_no_self_attention",
    "cloth_no_ffn",
    "cloth_no_all",
    "cloth_on_needle",
    "elastic_drag",
    "elastic_drop",
    "twist",
    "twist_force",
    "twist_force_override",
    "duck_drop",
    "hair_ball",
}

FORCE_TRAJECTORY_DATA = {
    "elastic_drag",
    "real_world",
    "twist_force",
    "real_world_override",
    "twist_force_override",
}

FORCE_POSITION_OVERRIDE_DATA = {
    "real_world_override",
    "twist_force_override",
}

TOPOLOGY_DEFAULT_SEARCH_RADIUS_BY_DATA = {
    "flag": 0.05,
    "cloth": 0.05,  # backward-compatible alias
    "elastic": 0.03,
    "cloth_on_needle": 0.005,
    "cloth_on_ball": 0.01,
    "cloth_no_self_attention": 0.01,
    "cloth_no_ffn": 0.01,
    "cloth_no_all": 0.01,
    "elastic_drag": 0.01,
    "elastic_drop": 0.01,
    "twist": 0.007,
    "twist_force": 0.007,
    "twist_force_override": 0.007,
    "duck_drop":0.004,
    "hair_ball":0.008
}

OBSTACLE_DATA_TYPES = {"fluid"} | OBSTACLE_NPZ_DATA

def _decode_array(arr_dict):
    """
    Decode an ndarray dictionary from msgpack into a NumPy array.
    """
    dtype = arr_dict[b'type']
    shape = arr_dict.get(b'shape', None)
    data = np.frombuffer(arr_dict[b'data'], dtype=dtype)
    if shape is not None:
        data = data.reshape(shape)
    return data

def _read_msgpack_zst(file_path):
    with open(file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(f.read())
    return msgpack.unpackb(decompressed, raw=False)

def load_from_fluid(
    base_path,
    sim_prefix,
    start_idx,
    end_idx,
):
    """
    Returns
    -------
    pos : (T, N, 3)
    vel : (T, N, 3)
    feats : (N, 2)   # [mass, viscosity]
    box : (Nb, 3)
    box_feats : (Nb, 3)
    """

    all_pos = []
    all_vel = []

    box = None
    box_feats = None
    feats = None

    for file_idx in range(start_idx, end_idx + 1):
        file_name = f"{sim_prefix}{file_idx:02d}.msgpack.zst"
        file_path = os.path.join(base_path, file_name)

        frames = _read_msgpack_zst(file_path)

        for frame_i, frame in enumerate(frames):

            # ---- Position and velocity are required for every frame. ----
            pos = _decode_array(frame["pos"])
            vel = _decode_array(frame["vel"])

            all_pos.append(pos)
            all_vel.append(vel)

            # ---- Read static features only from the first frame of the first file. ----
            if box is None and frame_i == 0 and file_idx == start_idx:

                box = _decode_array(frame["box"])
                box_feats = _decode_array(frame["box_normals"])

                m = _decode_array(frame["m"])               # (N,)
                viscosity = _decode_array(frame["viscosity"])  # (N,)

                feats = np.stack([m, viscosity], axis=-1)  # (N, 2)

    # ---- Stack frames into (T, N, 3). ----
    pos = np.stack(all_pos, axis=0)
    vel = np.stack(all_vel, axis=0)
    
    pos = torch.from_numpy(pos).float()
    vel = torch.from_numpy(vel).float()
    feats = torch.from_numpy(feats).float()
    box = torch.from_numpy(box).float()
    box_feats = torch.from_numpy(box_feats).float()
    return pos, vel, feats, box, box_feats

def normalize_feats(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """

    # Use only columns 4-6.
    young = feats[:, 3]
    density = feats[:, 4]
    friction = feats[:, 5]

    # ---- Young modulus (log scale) ----
    young_log = np.log10(young)

    # Widen the range to support extrapolation.
    young_min = np.log10(5e3)
    young_max = np.log10(5e5)

    young_norm = (young_log - young_min) / (young_max - young_min)

    # ---- Density ----
    density_min = 50.0
    density_max = 500.0

    density_norm = (density - density_min) / (density_max - density_min)

    # ---- Friction ----
    friction_min = 0.0
    friction_max = 1.0

    friction_norm = (friction - friction_min) / (friction_max - friction_min)

    feats_norm = np.stack(
        [young_norm, density_norm, friction_norm],
        axis=1
    )

    return feats_norm.astype(np.float32)

def normalize_feats_elastic(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """

    # Use only columns 4-6.
    young = feats[:, 4]
    poisson_ratio = feats[:, 5]

    # ---- Young modulus (log scale) ----
    young_log = np.log10(young)

    # Widen the range to support extrapolation.
    young_min = np.log10(5e3)
    young_max = np.log10(5e5)

    young_norm = (young_log - young_min) / (young_max - young_min)

    # ---- Friction ----
    poisson_ratio_min = 0.0
    poisson_ratio_max = 1.0

    poisson_ratio_norm = (poisson_ratio - poisson_ratio_min) / (poisson_ratio_max - poisson_ratio_min)

    feats_norm = np.stack(
        [young_norm, poisson_ratio_norm],
        axis=1
    )

    return feats_norm.astype(np.float32)

def normalize_feats_elastic_drag(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """
    return np.ones((feats.shape[0], 1), dtype=np.float32)

def normalize_feats_elastic_drop(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """

    # Use only columns 4-6.
    young = feats[:, -2]
    poisson_ratio = feats[:, -1]

    # ---- Young modulus (log scale) ----
    young_log = np.log10(young)

    # Widen the range to support extrapolation.
    young_min = np.log10(1e4)
    young_max = np.log10(5e4)

    young_norm = (young_log - young_min) / (young_max - young_min)

    # ---- Friction ----
    poisson_ratio_min = 0.0
    poisson_ratio_max = 1.0

    poisson_ratio_norm = (poisson_ratio - poisson_ratio_min) / (poisson_ratio_max - poisson_ratio_min)

    feats_norm = np.concatenate(
        [
            young_norm[:, None],
            poisson_ratio_norm[:, None],
        ],
        axis=1
    )

    return feats_norm.astype(np.float32)

def normalize_feats_sand(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """
    radius = (np.log10(np.clip(feats[:,3],1e-12, None))+2.414640)/0.111540
    friction = 2 * (feats[:,8] - 0.1)/1.0 - 1.0
    feats_norm = np.stack(
        [radius, friction],
        axis=1
    )

    return feats_norm.astype(np.float32)

def normalize_feats_non(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """
    young  = 0.2 * (feats[:,0] - 7) -1.0
    damp = 0.04 * (feats[:,2] - 50) -1.0
    feats_norm = np.stack(
        [young, damp],
        axis=1
    )

    return feats_norm.astype(np.float32)

def normalize_feats_cloth(feats):
    """
    feats: (N,6)
    return: (N,3) normalized features
    """

    # Use only columns 4-6.
    mass = feats[:, 0]
    young = feats[:, -2]
    density = feats[:, -1]

    # ---- Young modulus (log scale) ----
    young_log = np.log10(young)

    # Widen the range to support extrapolation.
    young_min = np.log10(5e3)
    young_max = np.log10(3e5)

    young_norm = (young_log - young_min) / (young_max - young_min)

    # ---- Friction ----
    density_min = 50
    density_max = 500

    density_norm = (density - density_min) / (density_max - density_min)

    mass_norm = (mass - np.mean(mass, axis=0)) / (np.std(mass, axis=0) + 1e-8)

    feats_norm = np.concatenate(
        [
            mass_norm[:, None],
            young_norm[:, None],
            density_norm[:, None],
        ],
        axis=1
    )

    return feats_norm.astype(np.float32)

def transform_particle_feats_by_data(data, feats):
    if data in {"cloth_on_ball", "cloth_no_self_attention", "cloth_no_ffn", "cloth_no_all"}:
        return normalize_feats_cloth(feats)
    if data in {"elastic"}:
        return normalize_feats_elastic(feats)
    if data in {"sand"}:
        return normalize_feats_sand(feats)
    if data in {"non", "non_obj", "non_bunny"}:
        return normalize_feats_non(feats)
    if data in {
        "fancy_fluid",
        "fancy_fluid_abl",
        "fancy_flulid_div",
        "fancy_fluid_div",
        "local_fluid",
    }:
        col_2 = feats[:, 2:3]
        return (col_2 - np.mean(col_2, axis=0)) / (np.std(col_2, axis=0) + 1e-8)
    if data == "cloth_on_needle":
        return feats[:, 4:7]
    if data == "elastic_drag":
        return normalize_feats_elastic_drag(feats)
    if data == "elastic_drop":
        col_0 = feats[:, 0:1]
        return (col_0 - np.mean(col_0, axis=0)) / (np.std(col_0, axis=0) + 1e-8)
    if data in {"real_world", "real_world_override", "duck_drop"}:
        return feats
    if data in {"twist", "twist_force", "twist_force_override", "hair_ball"}:
        col_0 = feats[:, 0:1]
        return (col_0 - np.mean(col_0, axis=0)) / (np.std(col_0, axis=0) + 1e-8)
    print("[Warning]: data return inaccurate feats.")
    return feats

def uses_topology_neighbors(data):
    return data in TOPOLOGY_ENABLED_DATA

def get_default_topology_search_radius(data):
    return TOPOLOGY_DEFAULT_SEARCH_RADIUS_BY_DATA.get(data, None)

def uses_force_trajectory(data):
    return data in FORCE_TRAJECTORY_DATA

def uses_force_position_override(data):
    return data in FORCE_POSITION_OVERRIDE_DATA

def is_obstacle_npz_data(data):
    return data in OBSTACLE_NPZ_DATA

def is_no_obstacle_npz_data(data):
    return data in NO_OBSTACLE_NPZ_DATA

def uses_obstacle_features(data):
    return data in OBSTACLE_DATA_TYPES

def infer_other_feats_channels(
    data_folder,
    data_type,
    data_list="train",
):
    """Infer particle feature channel count from feats.shape[-1]."""
    if data_folder is None:
        raise ValueError("infer_other_feats_channels requires data_folder.")

    list_path = os.path.join(data_folder, f"{data_list}.txt")
    if not os.path.exists(list_path):
        raise ValueError(f"Missing list file for channel inference: {list_path}")

    with open(list_path, "r") as f:
        name_list = [line.strip() for line in f if line.strip()]
    if len(name_list) == 0:
        raise ValueError(f"Empty list file for channel inference: {list_path}")

    if data_type == "fluid":
        data_path = os.path.join(data_folder, data_list)
        for name in name_list:
            try:
                sample = load_fluid_sample_by_data(
                    data_path=data_path,
                    sim_name=name,
                    start_idx=0,
                    end_idx=0,
                )
                feats = sample.get("feats", None)
                if feats is None:
                    continue
                if feats.dim() == 1:
                    return 1
                return int(feats.shape[-1])
            except Exception:
                continue
        raise ValueError("Failed to infer other_feats_channels from fluid data.")

    if data_type == "trajectory":
        for name in name_list:
            npz_path = os.path.join(data_folder, name, f"{name}.npz")
            pdb_path = os.path.join(data_folder, name, f"{name}.pdb")
            if not os.path.exists(npz_path) or not os.path.exists(pdb_path):
                continue
            try:
                sample = load_trajectory_sample_by_data(
                    npz_path=npz_path,
                    pdb_path=pdb_path,
                    frame_start=0,
                    frame_end=1,
                )
                feats = sample.get("feats", None)
                if feats is None:
                    continue
                if feats.dim() == 1:
                    return 1
                return int(feats.shape[-1])
            except Exception:
                continue
        raise ValueError("Failed to infer other_feats_channels from trajectory data.")

    if is_no_obstacle_npz_data(data_type) or is_obstacle_npz_data(data_type):
        for name in name_list:
            npz_path = os.path.join(data_folder, name, f"{name}.npz")
            if not os.path.exists(npz_path):
                continue
            try:
                sample = load_npz_sample_by_data(
                    data_type=data_type,
                    npz_path=npz_path,
                    frame_start=0,
                    frame_end=1,
                )
                feats = sample.get("feats", None)
                if feats is None:
                    continue
                if feats.dim() == 1:
                    return 1
                return int(feats.shape[-1])
            except Exception:
                continue
        raise ValueError(f"Failed to infer other_feats_channels from npz data type={data_type}.")

    raise ValueError(f"Unsupported data type for other_feats_channels inference: {data_type}")

def infer_obstacle_feats_channels(
    data_folder,
    data_type,
    data_list="train",
):
    """Infer obstacle feature channel count from dataset files."""
    if not uses_obstacle_features(data_type):
        return None
    if data_folder is None:
        raise ValueError("infer_obstacle_feats_channels requires data_folder.")

    list_path = os.path.join(data_folder, f"{data_list}.txt")
    if not os.path.exists(list_path):
        raise ValueError(f"Missing list file for channel inference: {list_path}")

    with open(list_path, "r") as f:
        name_list = [line.strip() for line in f if line.strip()]
    if len(name_list) == 0:
        raise ValueError(f"Empty list file for channel inference: {list_path}")

    if data_type == "fluid":
        data_path = os.path.join(data_folder, data_list)
        for name in name_list:
            try:
                sample = load_fluid_sample_by_data(
                    data_path=data_path,
                    sim_name=name,
                    start_idx=0,
                    end_idx=0,
                )
                box_feats = sample.get("box_feats", None)
                if box_feats is None:
                    continue
                if box_feats.dim() == 1:
                    return 1
                return int(box_feats.shape[-1])
            except Exception:
                continue
        raise ValueError("Failed to infer obstacle_feats_channels from fluid data.")

    if is_obstacle_npz_data(data_type):
        for name in name_list:
            npz_path = os.path.join(data_folder, name, f"{name}.npz")
            if not os.path.exists(npz_path):
                continue
            try:
                with np.load(npz_path) as data:
                    if "box_feats" not in data:
                        continue
                    box_feats = data["box_feats"]
                    if box_feats.ndim == 1:
                        return 1
                    return int(box_feats.shape[-1])
            except Exception:
                continue

    raise ValueError(f"Failed to infer obstacle_feats_channels from data type={data_type}.")

def load_fluid_sample_by_data(
    data_path,
    sim_name,
    start_idx=0,
    end_idx=15,
):
    positions, velocities, feats, box, box_feats = load_from_fluid(
        base_path=data_path,
        sim_prefix=sim_name,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    F, N, _ = positions.shape
    return {
        "positions": positions,
        "velocities": velocities,
        "num_frames": F,
        "num_atoms": N,
        "feats": feats,
        "box": box,
        "box_feats": box_feats,
        "topology_neighbors_index": None,
        "topology_neighbors_row_splits": None,
        "force_traj": None,
    }

def load_npz_sample_by_data(
    data_type,
    npz_path,
    frame_start=0,
    frame_end=100,
):
    with np.load(npz_path) as data:
        sample = {
            "positions": None,
            "velocities": None,
            "num_frames": 0,
            "num_atoms": 0,
            "feats": None,
            "box": None,
            "box_feats": None,
            "topology_neighbors_index": None,
            "topology_neighbors_row_splits": None,
            "force_traj": None,
        }

        positions = torch.from_numpy(data["position"][frame_start:frame_end]).float()
        velocities = torch.from_numpy(data["velocity"][frame_start:frame_end]).float()

        if data_type == "duck_drop":
            positions = torch.from_numpy(data["position"][::5][frame_start:frame_end]).float()
            velocities = torch.from_numpy(data["velocity"][::5][frame_start:frame_end]).float()

        F, N, _ = positions.shape
        feats = torch.from_numpy(transform_particle_feats_by_data(data_type, data["feats"])).float()

        topology_neighbors_index = None
        topology_neighbors_row_splits = None
        if uses_topology_neighbors(data_type):
            if "neighbors_index" in data and "neighbors_row_splits" in data:
                topology_neighbors_index = torch.from_numpy(data["neighbors_index"]).long()
                topology_neighbors_row_splits = torch.from_numpy(data["neighbors_row_splits"]).long()

        force_traj = None
        if uses_force_trajectory(data_type):
            if "force" in data:
                force_traj = torch.from_numpy(data["force"][frame_start:frame_end]).float()
            else:
                if uses_force_position_override(data_type):
                    raise KeyError(
                        f"data={data_type} requires npz['force'], but {os.path.basename(npz_path)} is missing it."
                    )
                print(
                    f"⚠️ data={data_type} expects npz['force'] but {os.path.basename(npz_path)} is missing it. "
                    "Fallback to config gravity."
                )
        if is_no_obstacle_npz_data(data_type):
            sample.update(
                positions=positions,
                velocities=velocities,
                num_frames=F,
                num_atoms=N,
                feats=feats,
                topology_neighbors_index=topology_neighbors_index,
                topology_neighbors_row_splits=topology_neighbors_row_splits,
                force_traj=force_traj,
            )
            return sample

        if is_obstacle_npz_data(data_type):
            box = torch.from_numpy(data["box"]).float()
            box_feats = torch.from_numpy(data["box_feats"]).float()
            sample.update(
                positions=positions,
                velocities=velocities,
                num_frames=F,
                num_atoms=N,
                feats=feats,
                box=box,
                box_feats=box_feats,
                topology_neighbors_index=topology_neighbors_index,
                topology_neighbors_row_splits=topology_neighbors_row_splits,
                force_traj=force_traj,
            )
            return sample

    raise ValueError(f"Unsupported npz data type for this loader: {data_type}")


def load_trajectory_sample_by_data(
    npz_path,
    pdb_path,
    frame_start=0,
    frame_end=100,
    return_topology=False,
):
    traj = md.load(pdb_path)
    masses_list = []
    type_idx_list = []

    for i, atom in enumerate(traj.topology.atoms):
        if atom.element is None:
            print(f"Warning: Atom {i} has unknown element. Using mass=1.0, type='X'")
            mass = 1.0
            atom_type = "X"
        else:
            mass = float(atom.element.mass)
            atom_type = atom.element.symbol

        masses_list.append(mass)
        type_idx_list.append(element_to_idx.get(atom_type, UNK_IDX))

    feats = torch.cat(
        [
            torch.tensor(masses_list, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(type_idx_list, dtype=torch.float32).unsqueeze(-1),
        ],
        dim=1,
    )

    with np.load(npz_path, mmap_mode="r") as data:
        positions = torch.from_numpy(data["position"][frame_start:frame_end]).float()
        velocities = torch.from_numpy(data["velocity"][frame_start:frame_end]).float()

    F, N, _ = positions.shape
    if feats.shape[0] != N:
        raise ValueError(
            f"Sample feature size mismatch: feats={feats.shape[0]} vs num_atoms={N} "
            f"for {os.path.basename(npz_path)}"
        )

    sample = {
        "positions": positions,
        "velocities": velocities,
        "num_frames": F,
        "num_atoms": N,
        "feats": feats,
        "box": None,
        "box_feats": None,
        "topology_neighbors_index": None,
        "topology_neighbors_row_splits": None,
        "force_traj": None,
    }
    if return_topology:
        sample["topology"] = traj.topology
    return sample


class TrajDataset(Dataset):
    """
    Trajectory dataset that returns sub-trajectories instead of frame pairs.

    Each item contains:
        pos_traj: [L, N, 3]
        vel_traj: [L, N, 3]
        dt: float  (frame skip)
        masses: [N]
        atom_types: [N]

    Modes:
        - "full": return ALL valid sub-trajectories from all samples
        - "random": every __getitem__ generates a random sub-trajectory
    """

    def __init__(self,
                 data_folder,
                 data_list,
                 mode="random",
                 data="fluid",
                 frame_start=0,
                 frame_end=100,
                 num_samples=2000):
        """
        Args:
            data_folder: folder containing *.pdb + *.npz files
            mode: "random" or "full"
            num_samples: used only under "random" mode
        """
        super().__init__()
        assert mode in ["random", "full"]

        self.mode = mode
        data_type = data
        self.num_samples = num_samples

        data_list_path = os.path.join(data_folder, f"{data_list}.txt")
        with open(data_list_path, "r") as f:
            name_list = [line.strip() for line in f if line.strip()]
            
        self.samples = []
        
        if data == "trajectory":
            for name in tqdm(name_list, desc="Loading samples"):
                pdb_path = os.path.join(data_folder, name, f"{name}.pdb")
                npz_path = os.path.join(data_folder, name, f"{name}.npz")
                if not os.path.exists(pdb_path):
                    print(f"⚠️ Missing PDB for {name}, skipping.")
                    continue

                sample = load_trajectory_sample_by_data(
                    npz_path=npz_path,
                    pdb_path=pdb_path,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
                self.samples.append({"name": name, **sample})
                    
        elif data == "fluid":
            for name in tqdm(name_list, desc="Loading fluid xyz"):
                data_path = os.path.join(data_folder, data_list)
                sample = load_fluid_sample_by_data(
                    data_path=data_path,
                    sim_name=name,
                    start_idx=0,
                    end_idx=15,
                )
                self.samples.append({"name": name, **sample})
        elif is_no_obstacle_npz_data(data_type):
            for name in tqdm(name_list, desc="Loading no-obstacle npz data"):
                npz_path = os.path.join(data_folder, name, f"{name}.npz")
                sample = load_npz_sample_by_data(
                    data_type=data_type,
                    npz_path=npz_path,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
                self.samples.append({"name": name, **sample})

        elif is_obstacle_npz_data(data_type):
            for name in tqdm(name_list, desc="Loading npz particle data"):
                npz_path = os.path.join(data_folder, name, f"{name}.npz")
                sample = load_npz_sample_by_data(
                    data_type=data_type,
                    npz_path=npz_path,
                    frame_start=frame_start,
                    frame_end=frame_end,
                )
                self.samples.append({"name": name, **sample})
        else:
            raise ValueError(f"Unsupported data type in TrajDataset: {data_type}")

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples loaded for data='{data}' from {data_list_path}."
            )

        print(f"Total Data: {len(self.samples)}")

    def __len__(self):
        if self.mode == "full":
            return len(self.samples)
        else:
            return self.num_samples
        
    def set_max_traj_len(self, max_traj_len):
        self.max_traj_len = max_traj_len

    def _sample_random(self):
        sample_info = random.choice(self.samples)
        return sample_info

    def __getitem__(self, sample_index):
        """
        sample_index: tuple (sample_idx, start, rollout, skip)
        """
        sample_idx, start, rollout, skip = sample_index
        sample_info = self.samples[sample_idx]

        frames = start + torch.arange(0, rollout * skip, skip)
        frames = frames.long()

        pos = sample_info["positions"][frames]   # [L, N, 3]
        vel = sample_info["velocities"][frames]  # [L, N, 3]
        feats = sample_info["feats"]
        box = sample_info.get("box", None)
        box_feats = sample_info.get("box_feats", None)
        topology_neighbors_index = sample_info.get("topology_neighbors_index", None)
        topology_neighbors_row_splits = sample_info.get("topology_neighbors_row_splits", None)
        force_traj = sample_info.get("force_traj", None)
        if force_traj is not None:
            force_traj = force_traj[frames]

        if topology_neighbors_index is not None and topology_neighbors_row_splits is not None:
            # Default PyTorch collation cannot stack None values, so use empty
            # placeholders for no-obstacle topology samples and restore them later.
            if box is None:
                box = pos.new_empty((0, pos.shape[-1]))
            if box_feats is None:
                feat_dim = feats.shape[-1] if feats.dim() > 1 else 1
                box_feats = feats.new_empty((0, feat_dim))
            if force_traj is not None:
                return (
                    pos,
                    vel,
                    feats,
                    box,
                    box_feats,
                    topology_neighbors_index,
                    topology_neighbors_row_splits,
                    force_traj,
                )
            return pos, vel, feats, box, box_feats, topology_neighbors_index, topology_neighbors_row_splits
        if box is not None and box_feats is not None:
            if force_traj is not None:
                return pos, vel, feats, box, box_feats, force_traj
            return pos, vel, feats, box, box_feats
        if force_traj is not None:
            return pos, vel, feats, force_traj
        return pos, vel, feats

class SubTrajectoryBatchSampler(Sampler):
    """
    Fully enumerates ALL sub-trajectories for each sample.
    Compatible with PyTorch Lightning + DDP.

    Every batch consists of sub-trajectories from ONE sample.

    A "sample index" is a tuple:
        (sample_idx, start_frame, rollout, skip)

    You control:
        rollout_range: list of possible sub-trajectory lengths
        skip_range: list of possible frame skips
    """

    def __init__(self,
                 dataset,
                 rollout=5,
                 skip_range=(1, 2, 3, 4, 5),
                 batch_size=4,
                 shuffle=True,
                 drop_last=False,
                 seed=42):

        self.dataset = dataset
        self.rollout = rollout
        self.skip_range = skip_range
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # Precompute all sample indices.
        self.all_samples = self._build_all_subtraj_indices()

        # Bucket sample indices by sample_idx.
        self.buckets = self._bucket_by_sample()

        # Store the complete batch list for the epoch.
        self._all_batches = []
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def set_rollout(self, new_rollout):
        self.rollout = new_rollout
        self.all_samples = self._build_all_subtraj_indices()
        self.buckets = self._bucket_by_sample()

    # ----------------------------------------------------
    # Build all subtrajectory indices.
    # ----------------------------------------------------
    def _build_all_subtraj_indices(self):
        all_samples = []  # Each item is (sample_idx, start, rollout, skip).

        for sample_idx, sample_info in enumerate(self.dataset.samples):
            num_frames = sample_info["num_frames"]

            for skip in self.skip_range:
                total_span = (self.rollout - 1) * skip + 1
                if total_span > num_frames:
                    continue

                    # Enumerate every valid start index.
                for start in range(0, num_frames - total_span + 1):
                    all_samples.append((sample_idx, start, self.rollout, skip))

        return all_samples

    # ----------------------------------------------------
    # Bucket samples by sample.
    # ----------------------------------------------------
    def _bucket_by_sample(self):
        """
        bucket key: sample_idx
        value: list of global sample indices in that bucket
        """
        buckets = {}

        for i, (sample_idx, start, rollout, skip) in enumerate(self.all_samples):

            if sample_idx not in buckets:
                buckets[sample_idx] = []

            buckets[sample_idx].append(i)

        return buckets

    # ----------------------------------------------------
    # Build batches for each epoch on rank 0.
    # ----------------------------------------------------
    def _make_all_batches(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches = []

        for sample_idx, indices in self.buckets.items():
            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)

            for i in range(0, len(idxs), self.batch_size):
                idx_batch = idxs[i:i+self.batch_size]
                if len(idx_batch) == self.batch_size or not self.drop_last:
                    # convert to tuples
                    batch = [self.all_samples[idx] for idx in idx_batch]
                    all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        self._all_batches = all_batches

    # ----------------------------------------------------
    # Distribute batches across ranks.
    # ----------------------------------------------------
    def __iter__(self):
        # Generate all batches for this epoch.
        self._make_all_batches()

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        else:
            rank = 0
            world = 1
        total = len(self._all_batches)
        total = (total // world) * world
        batches = self._all_batches[:total]

        for i in range(rank, len(batches), world):
            yield batches[i]

    # ----------------------------------------------------
    def __len__(self):
        total = 0
        for idxs in self.buckets.values():
            size = len(idxs)
            if self.drop_last:
                total += size // self.batch_size
            else:
                total += math.ceil(size / self.batch_size)

        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
        else:
            world = 1

        # Keep __len__ consistent with __iter__, which truncates to a world-size multiple.
        return total // world

if __name__ == "__main__":
    # quick test
    TrajDataset(
                data_folder="/workspace/liwen/TransformerMD/flag",
                data_list="train",
                mode="full",
                num_samples=500,
                data="cloth"
            )
