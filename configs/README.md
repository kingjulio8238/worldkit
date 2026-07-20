# Hydra configs

Hydra YAML configuration for training and inference. Entry-point scripts under `scripts/` load
these via `@hydra.main(config_path="../configs")`, so any key can be overridden on the command line
as `key=value`.

```
configs/
  train_codec.yaml          # codec training (RAEv2 temporal-downsampling)
  train_world_model.yaml    # latent world-model training
  eval_world_model.yaml     # offline world-model evaluation
  model/                    # model architectures (codec, latent world model, multiplayer wrapper)
  dataset/                  # dataset source (train/test splits, n_players, target fps)
  actions/                  # action vocabulary (9-key DEFAULT_RL_KEYS, target fps)
```

## Entry-point configs

- **`train_codec.yaml`** composes `model: raev2_codec_tdown` and `dataset: rocket_league`, plus the
  `run`, `wandb`, `dataloader`, `validation`, and `optim` blocks for codec reconstruction training.
- **`train_world_model.yaml`** composes `model: latent_world_model` and `dataset: rocket_league`,
  and adds a `world_model_metrics` block (rollout DINO/latent drift and Frechet curves) run every
  `validation.downstream_val_every`.
- **`eval_world_model.yaml`** is the offline-evaluation entry point. It carries only the
  `world_model_metrics` block; the checkpoint supplies its own model and dataset config, so this
  file just configures how the eval is run (set `checkpoint` and `output_dir` at eval time).
- **`serve_psd_2step.yaml`** is the validated fast-serving preset: the 2-step `mira-mini-psd`
  checkpoint + the full inference stack (compile + precomputed RoPE + CUDA graphs) → ~24.7 ms/
  latent-frame, ~7× the released 10-step base at matched quality. Its `inference` block maps 1:1 onto
  `WorldModelInferenceConfig` (see the file header for the load snippet and `docs/optimization_plan.md`).

## `model/`

- **`raev2_codec_tdown.yaml`** — RAEv2 codec with 2× temporal downsampling: a frozen DINOv3-L/16
  backbone with layer aggregation and a strided-conv bottleneck feeding a ViT video decoder, with
  the L1 + LPIPS + DINO-latent-consistency loss weights.
- **`latent_world_model.yaml`** — single-player latent world model: a frozen codec (loaded from
  `codec_checkpoint`) plus an action-conditioned flow-matching diffusion transformer over the
  codec's latent grid.
- **`multi_wrapper_world_model.yaml`** — multiplayer world model: an inner `LatentWorldModel` that
  processes `n_players` per-player clips tiled into one vertically-stacked frame. A single-player
  checkpoint warm-starts into it via `run.finetune_from`.
- **`latent_world_model/1b.yaml`** — the 1B-parameter transformer size (hidden dim, heads, layers),
  selected by the package-override `latent_world_model@architecture.config: 1b`.

## `dataset/` and `actions/`

`dataset/rocket_league.yaml` points `train_index`/`test_index` at the local split directories (or
fetches them from the Hub) and sets `n_players` (1 for the codec, overridden to 4 for the
multiplayer world model) and `target_fps`. It pulls in the action vocabulary with
`/actions@actions: rocket_league`.

`actions/rocket_league.yaml` lists the 9-key release vocabulary (`DEFAULT_RL_KEYS`) and the action
sample rate `target_fps`.

## Interpolation

Configs use Hydra/OmegaConf `${...}` interpolation to keep shared values in one place:

- `actions: ${dataset.actions}` lifts the dataset's action vocabulary to the top level for the
  loader.
- the top-level `run.compile` toggle threads into the codec's DINO compilation
  (`compile_dino: ${run.compile}`) and into the world-model metrics block (`compile: ${run.compile}`).
- the codec decoder reads encoder values it must match (e.g. `latent_dim: ${..encoder.latent_dim}`,
  `bottleneck.stride: ${...encoder.bottleneck.stride}`).

The world-model architecture size is selected via package-override syntax
(`latent_world_model@architecture.config`).
