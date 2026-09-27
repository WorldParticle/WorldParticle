# WorldParticle

WorldParticle is a particle trajectory model built around local continuous convolutions, iterative `k`-partite token merging, and a super-particle decoder. The model supports both single-item and multi-item batches. The supplied inference CLI rolls out one sample at a time and predicts positions and velocities autoregressively.

The main entry points are:

- `config.yaml`: model, dataset, and rollout settings.
- `TrajDataset.py`: sample data loading.
- `trajectory_model.py`: model construction and forward computation.
- `inference.py` / `inference.bash`: autoregressive rollout inference.
- `models/particle_network_cross_attn_feat.py`: active model implementation.

## Environment

The provided `environment.yaml` contains the Python dependencies, but leaves PyTorch and FlashAttention commented out so that the CUDA-specific builds can be installed explicitly. FlashAttention is required for the intended inference speed.

```bash
conda env create -f environment.yaml
conda activate WorldParticle

python -m pip install --upgrade pip
python -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
python -m pip install flash-attn==2.8.3 --no-build-isolation
export ATTN_IMPL=flash_attn
```

Set `ATTN_IMPL=flash_attn` before inference. FlashAttention is the required runtime backend for this project.

Open3D is required by the continuous-convolution neighbor search. Verify the core runtime with:

```bash
python - <<'PY'
import open3d
import torch
import flash_attn

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("open3d:", open3d.__version__)
print("flash_attn:", flash_attn.__version__)
PY
```

## Data Layout

The dataset root is passed to inference as `--data_dir`. Samples are selected by name with `--sample`.

The loader selected by `config.yaml:data` determines the remaining layout.

### NPZ Particle Data

A complete NPZ example is `elastic_drag`, because it uses obstacle features, a topology graph, and an external-force trajectory. Set:

```yaml
data: elastic_drag
```

The expected layout is:

```text
DATA_ROOT/
  sample_001/
    sample_001.npz
  sample_002/
    sample_002.npz
```

For the complete `elastic_drag` path, each NPZ contains:

- `position`: `[T, N, 3]` particle positions.
- `velocity`: `[T, N, 3]` particle velocities.
- `feats`: `[N, C]` raw per-particle features. The `elastic_drag` transform produces the active per-particle feature channel used by the model.
- `box`: `[M, 3]` obstacle or boundary points.
- `box_feats`: `[M, Cb]` obstacle features, commonly boundary normals. The model infers `Cb` from the data.
- `neighbors_index`: `[E]` CSR topology-neighbor indices.
- `neighbors_row_splits`: `[N + 1]` CSR row splits delimiting the topology neighbors of every particle.
- `force`: `[T, N, 3]` external force applied during physics integration.

`elastic_drag` belongs to the obstacle, topology-enabled, and force-trajectory data groups in `TrajDataset.py`. Providing all arrays above exercises the complete local-feature path: fluid-fluid continuous convolution, fluid-obstacle continuous convolution, topology convolution, and force-aware integration. The topology graph and obstacle tensors are static for a sample, while position, velocity, and force are indexed by frame.

Other NPZ data types may omit `box`/`box_feats`, topology arrays, or `force` when their data-group rules do not use them. The checked-in `config.yaml` currently uses `fancy_fluid`; change its `data` field and data root when running the `elastic_drag` example.

`frame_start` and `frame_end` select the frame window loaded from NPZ particle datasets.

### Trajectory Data

Set `data: trajectory` for PDB-backed trajectories:

```text
DATA_ROOT/
  sample_001/
    sample_001.pdb
    sample_001.npz
```

The NPZ contains `position` and `velocity`. Particle mass and element-index features are derived from the PDB topology. The trajectory loader first applies the configured `frame_start:frame_end` window; inference then applies `--start` and `--end` relative to that window.

### Raw Fluid Msgpack Data

Set `data: fluid` to load compressed msgpack frames:

```text
DATA_ROOT/
  train/
    sample_00100.msgpack.zst
    ...
    sample_00115.msgpack.zst
```

The `--sample` argument is a filename prefix. For example, `--sample sample_001` loads files `sample_00100.msgpack.zst` through `sample_00115.msgpack.zst`. The `train/` directory name is retained for compatibility with the raw fluid data layout.

## Model Path

One model step performs:

1. Physics integration using `gravity`, `timestep`, and an optional external force.
2. Batched radius searches and continuous convolutions for local particle, obstacle, and optional topology features.
3. Iterative super-particle construction with `k_partite_soft_matching`.
4. Particle-to-super-particle decoding with RoPE attention.
5. Position and, when `velhead: true`, velocity corrections.

The token merge path uses the `k`-partite implementation.

For multi-item batches, radius-search row splits prevent cross-sample neighbors. Tokenization applies the same single-item merge path to every batch item and concatenates the results, preserving the token count and ordering used by batch size one.

With `k_partite: k`, each merge iteration reduces the token count to approximately `floor(N / k)`. Choose `slot_attn_iters` so that the particle count remains nonzero through every iteration.

## Configuration

### Runtime And Data

| Parameter | Meaning |
| --- | --- |
| `device` | Preferred inference device. |
| `data` | Loader and model data type. The checked-in config uses `fancy_fluid`. |
| `frame_start`, `frame_end` | Frame window for NPZ particle datasets. |

### Physics And Local Features

| Parameter | Meaning |
| --- | --- |
| `gravity` | Constant acceleration vector. |
| `timestep` | Integration time step. |
| `particle_radius` | Base particle radius. |
| `radius_scale` | Multiplier used to derive neighbor-search filter extents. |
| `kernel_size` | Continuous-convolution kernel resolution. |
| `cconv_embedding_dim` | Per-branch local feature dimension. |
| `use_topology_conv` | `null` uses the built-in data-type rule; a boolean overrides it. |
| `topology_search_radius` | `null` uses the data-type default; a number overrides it. |
| `topology_blend_alpha` | When topology is enabled, `null` concatenates fluid and topology branches, `0` is topology-only, and `1` is fluid-only. |

`coordinate_mapping`, `interpolation`, and `use_window` exist in `config.yaml`, but the `TrajectoryModel` wrapper does not forward them into the model constructor. The active path therefore uses the model defaults: `ball_to_cube_volume_preserving`, `linear`, and `true`.

### Tokenizer And Decoder

| Parameter | Meaning |
| --- | --- |
| `slot_compression_ratio` | Legacy constructor value; it does not determine the current token count. |
| `slot_attn_iters` | Number of iterative `k`-partite merge layers. |
| `slot_attn_heads` | Attention heads in each token merge layer. |
| `k_partite` | Merge partition factor; must be at least `2`. |
| `particle_position_rope_dim` | 3D RoPE dimension; must be divisible by `6`. |
| `num_decoder_layers` | Number of super-particle decoder layers. |
| `decoder_attn_heads` | Decoder attention heads. |
| `decoder_attn_ffn_hidden_dim` | Decoder FFN hidden dimension. |
| `decoder_attn_dropout` | Decoder attention and FFN dropout. The tokenizer currently uses its constructor default of `0.1`. |
| `output_hidden_dim` | Output MLP hidden dimension. |
| `output_layers` | Output MLP depth. |
| `velhead` | Enables separate learned position and velocity corrections. |
| `verbose` | Enables timing diagnostics and debugger breaks inside the model. Keep this `false` for normal runs. |

The combined feature dimension is `3 * cconv_embedding_dim` for obstacle data and `local_fluid`, and `2 * cconv_embedding_dim` for other no-obstacle data. It must be divisible by both `slot_attn_heads` and `decoder_attn_heads`. The current implementation also enforces:

```text
particle_position_rope_dim % 6 == 0
particle_position_rope_dim <= 3 * cconv_embedding_dim / decoder_attn_heads
```

## Inference

`inference.bash` is a batch launcher template. Set `DATA_DIR`, `FOLD_DIR`, the actual checkpoint filename, rollout bounds, and sample names. Point `CHECKPOINT` to an existing checkpoint and use its accompanying configuration.

Equivalent direct command:

```bash
ATTN_IMPL=flash_attn CUDA_VISIBLE_DEVICES=0 python inference.py \
    --data_dir /path/to/data \
    --sample 0 \
    --config /path/to/checkpoints/run_name/config.yaml \
    --checkpoint /path/to/checkpoints/run_name/final.ckpt \
    --num_steps 199 \
    --dt 1 \
    --output_dir /path/to/checkpoints/run_name \
    --start 0 \
    --end 200
```

After editing the launcher, run it from the repository root:

```bash
bash inference.bash
```

Important inference behavior:

- The initial position and velocity come from frame `start` of the loaded `[start:end]` slice.
- `num_steps` controls the number of autoregressive predictions and may exceed the available ground-truth length.
- The model integration step comes from `config.yaml:timestep`. The `--dt` argument samples both predictions and ground truth when computing metrics and is used in XTC output naming; it does not override the model timestep.
- Feature channel counts are inferred from the loaded sample before model construction.
- `trajectory` data additionally writes an XTC trajectory using the PDB topology.

Predicted NPZ files are written to:

```text
OUTPUT_DIR/outcomes/npz/<sample>/<sample>_<START>_<END>.npz
```

They contain:

- `position`: predicted positions with the initial frame first.
- `velocity`: predicted velocities with the initial frame first.

When ground truth is available, inference also prints position and velocity MSE for the overlapping frames.
