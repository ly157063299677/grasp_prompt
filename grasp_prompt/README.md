# Target-Guided Grasp Prompt Diffusion

Code for object-side grasp prompt generation under target contact-part and
target-joint constraints. The implementation follows the method sections of the
paper: AffordPose-style label construction, conditional diffusion candidate
generation, TDA/CDCI denoising blocks, and target-guided candidate ranking with
spatial NMS.

## What Is Included

```text
configs/
  default.yaml                    Base configuration.
  debug_cpu.yaml                  Small CPU/debug run.
  affordpose_main.yaml            Main training configuration.
  affordpose_large.yaml           Larger-capacity setting.
  ablation_*.yaml                 Component-removal settings.
  affordpose_schema.md            Processed sample format.
examples/
  manifest_example.json           Manifest layout for data conversion.

scripts/
  prepare_affordpose.py           Convert raw/manifested data to npz samples.
  check_dataset.py                Validate shapes, masks, labels, and leakage.
  make_splits.py                  Object-level train/val/test split utility.
  train.py                        Training entry point with checkpoint resume.
  evaluate.py                     Offline metrics for paper-style reporting.
  infer.py                        Single-sample prompt generation.
  summarize_runs.py               Collect metric JSON files into a CSV table.
  make_ablation_commands.py       Print train/eval commands for ablations.
  make_part_joint_template.py      Create an editable part-to-joint CSV.
  make_toy_data.py                Small synthetic data generator for debugging.

src/grasp_prompt/
  data/                           Sample loading and label construction.
  models/                         Condition encoder, denoiser, selector.
  metrics.py                      Hit@1, Cover@K, MPD, CPA, AUC, JCS, CPS.
  baselines.py                    Random-valid and affordance-center baselines.
  training.py                     Dataloaders and checkpoint helpers.
```

## Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

Install the PyTorch build that matches the local CUDA driver before long GPU
runs. The rest of the package is standard Python and NumPy/PyTorch code.

## Data Format

Training uses one `.npz` file per interaction sample. Required fields are listed
in [configs/affordpose_schema.md](configs/affordpose_schema.md). The important
arrays are:

- object point cloud, normals, affordance labels, and valid-region mask;
- MANO or hand-state vectors and hand joint positions;
- hand-object relative state and task id;
- target contact-part vector and target joint mask;
- clustered prompt-region centers, normals, dominant part labels, and scales.

Prepare data from a manifest:

```bash
python scripts/prepare_affordpose.py --manifest data/raw_manifest.json --out data/affordpose_npz
python scripts/check_dataset.py --data data/affordpose_npz
```

See `examples/manifest_example.json` for the expected manifest structure.

If samples are first exported into a flat directory, create object-level splits:

```bash
python scripts/make_splits.py --input data/flat_npz --out data/affordpose_npz --group-key object_id
```

## Training

```bash
python scripts/train.py --config configs/affordpose_main.yaml --data data/affordpose_npz --run-dir runs/main
```

Resume from a checkpoint:

```bash
python scripts/train.py --config configs/affordpose_main.yaml --data data/affordpose_npz --run-dir runs/main --resume runs/main/checkpoint_last.pt
```

Run evaluation and save metrics:

```bash
python scripts/evaluate.py --config configs/affordpose_main.yaml --data data/affordpose_npz --checkpoint runs/main/checkpoint_best.pt --split test --out runs/main/metrics_test.json
```

Summarize several runs:

```bash
python scripts/summarize_runs.py runs/main runs/ablation_no_tda --out runs/summary.csv
```

## Ablations

The paper-style component removals are represented as separate config files:

- `configs/ablation_no_valid_mask.yaml`
- `configs/ablation_no_target_condition.yaml`
- `configs/ablation_no_tda.yaml`
- `configs/ablation_no_cdci.yaml`
- `configs/ablation_no_confidence.yaml`
- `configs/ablation_no_joint_guidance.yaml`
- `configs/ablation_no_snms.yaml`

To print a command list:

```bash
python scripts/make_ablation_commands.py --data data/affordpose_npz --runs runs
```

## Debug Run

The synthetic generator is only a pipeline check. It is useful after changing
the model or data loader, but it is not used for reported results.

```bash
python scripts/make_toy_data.py --out data/debug_npz --num-samples 64
python scripts/check_dataset.py --data data/debug_npz
python scripts/train.py --config configs/debug_cpu.yaml --data data/debug_npz --run-dir runs/debug_cpu --device cpu
```

## Reproducibility Notes

The code expects the study-specific contact-part definitions to be fixed before
final runs:

- the task-to-affordance mapping used for valid-region masks;
- MANO vertex groups for the hand/contact-part categories;
- the contact-part-to-joint association matrix;
- object-instance split files for the in-distribution and unseen-object
  protocols.

If `data.part_to_joint_path` is omitted, the package builds a minimal contiguous
matrix for debug and unit-test runs. Reported experiments should pass the study
matrix through `data.part_to_joint_path` in the config.

An editable CSV template can be created with:

```bash
python scripts/make_part_joint_template.py --out configs/part_to_joint_template.csv --with-header
```
