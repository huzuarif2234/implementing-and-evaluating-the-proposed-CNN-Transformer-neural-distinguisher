# CNN–Transformer Neural Distinguishers for Simeck and Simon

Implementation and evaluation of hybrid **CNN–Transformer neural distinguishers** for reduced-round **Simeck** and **Simon** block ciphers in a related-key setting.

The repository provides Basic and Enhanced implementations for 13 cipher variants. Each experiment script contains its cipher implementation, synthetic data generator, neural network, training routine, evaluation metrics, and an optional genetic algorithm (GA) pipeline for differential selection.

[Repository](https://github.com/huzuarif2234/implementing-and-evaluating-the-proposed-CNN-Transformer-neural-distinguisher) · [Report an issue](https://github.com/huzuarif2234/implementing-and-evaluating-the-proposed-CNN-Transformer-neural-distinguisher/issues)

## Contents

- [Overview](#overview)
- [Repository organization](#repository-organization)
- [Installation](#installation)
- [Quick start](#quick-start)
- [GA-based training pipeline](#ga-based-training-pipeline)
- [Configuration](#configuration)
- [Outputs and evaluation](#outputs-and-evaluation)
- [Included checkpoints](#included-checkpoints)
- [Reproducibility and implementation notes](#reproducibility-and-implementation-notes)
- [Citation and license](#citation-and-license)

## Overview

The experiments study whether a neural network can learn distinguishing patterns from ciphertext-derived features generated under selected plaintext and key differences. In the related-key setting, paired inputs use relations of the form:

$$P' = P \oplus \Delta P, \qquad K' = K \oplus \Delta K.$$

The model combines convolutional feature extraction, Transformer attention, residual convolutional blocks, and a sigmoid classification head. Several implementations also include learned positional embeddings and squeeze-and-excitation blocks. Architecture details and training defaults vary between scripts.

### Basic and Enhanced experiments

| Aspect | Basic | Enhanced |
| --- | --- | --- |
| Positive class | Samples generated using a selected plaintext/key differential | Samples drawn from a configured set of positive differentials |
| Negative class | Samples constructed using independently random plaintext/key pairs | Samples drawn from a separate set of negative differentials |
| Experimental question | Can the model separate the selected differential distribution from the random-pair construction? | Can the model separate two configured differential classes? |
| Differential selection | Built-in configurations or optional GA pipeline | Built-in positive/negative configurations, manual configurations, or optional GA pipeline |

**These are different classification tasks.** An Enhanced accuracy should not be interpreted as a directly comparable improvement over a Basic accuracy without accounting for the different negative-class construction.

Training and validation samples are generated locally; no external dataset download is required. Distinguishing reduced-round distributions does not, by itself, demonstrate full-round key recovery or a break of the complete cipher.

## Repository organization

The top-level directories are `Basic/` and `Enhanced/`. Each contains 13 numbered cipher directories with a standalone Python script and saved `.h5` checkpoints.

Variant names use **block size / key size**, in bits.

| Cipher | Basic directory | Enhanced directory |
| --- | --- | --- |
| Simeck32/64 | `Basic/1. simeck3264/` | `Enhanced/1. simeck3264/` |
| Simeck48/96 | `Basic/2. simeck4896/` | `Enhanced/2. simeck4896/` |
| Simeck64/128 | `Basic/3. simeck64128/` | `Enhanced/3. simck64128/` |
| Simon32/64 | `Basic/4. simon3264/` | `Enhanced/4. simon3264/` |
| Simon48/96 | `Basic/5. simon4896/` | `Enhanced/5. simon4896/` |
| Simon64/128 | `Basic/6. simon64128/` | `Enhanced/6. simon64128/` |
| Simon128/256 | `Basic/7. simon128256/` | `Enhanced/7. simon128256/` |
| Simon48/72 | `Basic/8. simon4872/` | `Enhanced/8. simon 4872/` |
| Simon64/96 | `Basic/9. simon6496/` | `Enhanced/9. simon6496/` |
| Simon96/144 | `Basic/10. simon96144/` | `Enhanced/10. simon96144/` |
| Simon128/192 | `Basic/11. simon128192/` | `Enhanced/11. simon128192/` |
| Simon96/96 | `Basic/12. simon9696/` | `Enhanced/12. simon9696/` |
| Simon128/128 | `Basic/13. simon128128/` | `Enhanced/13. simon128128/` |

Basic filenames follow `<variant>_cnn_transformer.py`. Enhanced filenames generally follow `<variant>_enhanced_cnn_transformer.py`, with two existing exceptions:

- Simeck32/64: `simeck3264_enhanced_cnn_transforme.py` — the final `r` is absent.
- Simon128/256: `simon128256_cnn_transformer.py` — the filename does not contain `enhanced`.

Use the directory and filename spellings shown above. Quote paths because directory names contain spaces.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/huzuarif2234/implementing-and-evaluating-the-proposed-CNN-Transformer-neural-distinguisher.git
cd implementing-and-evaluating-the-proposed-CNN-Transformer-neural-distinguisher
```

The repository includes binary model checkpoints, so cloning also downloads those files.

### 2. Create a Python environment

Use a Python version supported by your selected TensorFlow release.

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy tensorflow matplotlib h5py
```

The scripts use NumPy, TensorFlow/Keras, and Matplotlib; HDF5 support is needed for `.h5` checkpoints. The repository currently provides no dependency lockfile or tested version matrix. These commands install the required packages, but do not recreate the original training environment. Check your TensorFlow/Keras compatibility before launching a large experiment.

## Quick start

Run the following commands from the repository root. Training examples use Bash line continuations; in PowerShell, enter each command on one line or use PowerShell's continuation syntax.

### Inspect available options

```bash
python "Basic/1. simeck3264/simeck3264_cnn_transformer.py" --help
```

Each script has its own argument parser. Consult its `--help` output before transferring options between variants.

### Run the built-in sanity checks

```bash
python "Basic/1. simeck3264/simeck3264_cnn_transformer.py" --sanity_only
python "Enhanced/1. simeck3264/simeck3264_enhanced_cnn_transforme.py" --sanity_only
```

These invoke the scripts' cipher/data checks and exit before training. Internal consistency checks are not a substitute for independent validation against authoritative cipher test vectors.

### Start a small Basic experiment

```bash
python "Basic/1. simeck3264/simeck3264_cnn_transformer.py" \
  --rounds 13 \
  --epochs 2 \
  --n_train 20000 \
  --n_val 2000 \
  --batch_size 256 \
  --output_dir "results/basic_simeck3264_smoke"
```

### Start a small Enhanced experiment

```bash
python "Enhanced/1. simeck3264/simeck3264_enhanced_cnn_transforme.py" \
  --rounds 13 \
  --epochs 2 \
  --n_train 20000 \
  --n_val 2000 \
  --batch_size 256 \
  --output_dir "results/enhanced_simeck3264_smoke"
```

These small runs are intended to check the local training workflow. They are not sufficient to reproduce research-scale accuracy.

### Train multiple round configurations

```bash
python "Basic/1. simeck3264/simeck3264_cnn_transformer.py" \
  --rounds 13 14 \
  --epochs 30 \
  --n_train 20000000 \
  --n_val 2000000 \
  --batch_size 30000 \
  --output_dir "results/basic_simeck3264_full"
```

The full defaults are resource intensive. For Basic Simeck32/64, 20 million samples with 1,024 `uint8` features require approximately **20.48 GB for the training feature array alone**. Validation data, temporary arrays, framework copies, and model activations require additional memory. Begin with small sample counts and batch sizes, then scale to your hardware.

## GA-based training pipeline

Adding `--use_pipeline` enables an iterative process that searches for differentials, trains a model, checks performance thresholds, and refines the experiment within an iteration budget.

The following example uses a reduced search and training budget:

```bash
python "Basic/1. simeck3264/simeck3264_cnn_transformer.py" \
  --rounds 13 \
  --use_pipeline \
  --max_iterations 1 \
  --ga_pop_size 20 \
  --ga_generations 2 \
  --ga_n_fitness 1024 \
  --threshold_acc 0.60 \
  --epochs 2 \
  --n_train 20000 \
  --n_val 2000 \
  --batch_size 256 \
  --output_dir "results/basic_simeck3264_ga_smoke"
```

Here, `0.60` is an illustrative stopping threshold, not a claimed result. Reaching a threshold is not guaranteed. Multiple pipeline iterations can increase the data size or architecture size, depending on the script's refinement logic.

## Configuration

| Option | Purpose |
| --- | --- |
| `--rounds` | One or more reduced-round configurations |
| `--epochs` | Number of training epochs |
| `--n_train`, `--n_val` | Training and validation sample counts |
| `--batch_size` | Samples per training batch |
| `--s_groups` | Number of feature groups per sample |
| `--depth`, `--num_filters` | Convolutional architecture controls |
| `--d1`, `--d2` | Dense-layer widths |
| `--num_heads` | Attention heads |
| `--ff_dim` | Transformer feed-forward width |
| `--num_transformer_blocks` | Number of Transformer blocks |
| `--dropout`, `--reg_param` | Dropout and regularization settings |
| `--high_lr`, `--low_lr`, `--lr_epoch` | Cyclic learning-rate settings |
| `--output_dir` | Destination for generated results |
| `--sanity_only` | Run built-in checks without training |
| `--use_pipeline`, `--max_iterations` | Enable and bound iterative GA/training refinement |
| `--ga_pop_size`, `--ga_generations`, `--ga_n_fitness` | GA search budget |
| `--threshold_acc`, `--threshold_tpr`, `--threshold_tnr` | Pipeline performance thresholds |

Most scripts default to 20 million training samples, 2 million validation samples, and a batch size of 30,000. Other defaults vary by implementation.

Enhanced scripts also expose `--pos_diffs`, `--neg_diffs`, `--pos_key_diffs`, and `--neg_key_diffs` for manual differential configurations. Supply all four together, use the format described by the selected script, and match key-word counts to the cipher variant. Rounds absent from the built-in configuration may be skipped unless the required custom differences are provided.

## Outputs and evaluation

Training produces artifacts under `--output_dir`:

| Artifact | Content |
| --- | --- |
| `.h5` checkpoint | Model saved when validation accuracy improves |
| `.png` history plot | Training/validation accuracy and loss curves |
| `.txt` result summary | Configuration and reported validation metrics |
| Console output | Sanity checks, model summary, epoch logs, and final results |

For the Basic Simeck32/64 example at 13 rounds, the output names are `best_13r.h5`, `history_13r.png`, and `results_13r.txt`. Other scripts may include differential tags in filenames. Use a distinct output directory for each experiment to avoid overwriting prior results.

The principal classification metrics are:

$$\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN},\qquad
\mathrm{TPR}=\frac{TP}{TP+FN},\qquad
\mathrm{TNR}=\frac{TN}{TN+FP}.$$

The training routines select the epoch with the highest validation accuracy and report its associated metrics. These are **validation results**, not measurements on a separate final test set. For a research comparison, evaluate the selected checkpoint on fresh held-out samples after model selection.

Values named `expected_acc`, `expected_tpr`, or `expected_tnr` in source configuration tables are reference targets. They should not be reported as newly reproduced experimental results.

## Included checkpoints

The following inventory records round numbers appearing in the committed checkpoint filenames. It does not establish their measured accuracy or compatibility with the current script defaults.

| Cipher | Basic checkpoint rounds | Enhanced checkpoint rounds |
| --- | --- | --- |
| Simeck32/64 | 14 | 13, 14, 15 |
| Simeck48/96 | 15, 16, 17, 18, 19 | 18, 19 |
| Simeck64/128 | 21, 22 | 21, 22 |
| Simon32/64 | 12, 13, 14 | 11, 12, 13 |
| Simon48/96 | 12, 13, 14 | 13 |
| Simon64/128 | 13, 14 | 13, 14 |
| Simon128/256 | 15, 16, 17 | 16, 17 |
| Simon48/72 | 11, 12 | 12 |
| Simon64/96 | 11, 12, 13 | 12, 13 |
| Simon96/144 | 13, 14 | 13, 14 |
| Simon128/192 | 14, 15, 16 | 15 |
| Simon96/96 | 12, 13 | 12, 13 |
| Simon128/128 | 13, 14, 15 | 14, 15 |

The command-line interfaces do not provide a dedicated evaluation-only or checkpoint-resume option. Running a training command creates a new model; it does not automatically evaluate the adjacent checkpoint.

Before evaluating a saved model, establish its original architecture, feature layout, group count, cipher rounds, plaintext/key differences, class definitions, and TensorFlow/Keras environment. Existing checkpoint names and current defaults do not always match. For example, the Enhanced Simon48/96 script defaults to rounds 18 and 19, while its committed checkpoint is named for round 13.

## Reproducibility and implementation notes

- **Record the experiment environment.** Save the Git commit, package versions, hardware, full command, differential configuration, and output files. Run `git rev-parse HEAD` and `python -m pip freeze` to collect the code and dependency versions.
- **Random seeds do not control all data generation.** The scripts use `os.urandom` for plaintext/key material. Setting only a NumPy or TensorFlow seed will not reproduce identical datasets.
- **Read executable code alongside comments.** Some headers describe settings that differ from the implementation. For example, Enhanced Simeck32/64 compiles with binary cross-entropy despite its header mentioning MSE; other scripts use MSE.
- **Validate cipher conformance separately.** Built-in checks vary by script. An encrypt/decrypt round trip establishes internal consistency, but cannot alone establish agreement with the published cipher specification.
- **Check framework compatibility.** Some scripts use version-sensitive Keras APIs, including `get_shape()` on model weights. If model construction, parameter counting, or HDF5 loading fails, identify a compatible environment before scaling the experiment.
- **Keep evaluation protocols comparable.** Record class definitions and differential sets, reserve a final test set, and repeat experiments when reporting statistical performance.

This README documents the source and file inventory at commit `2bb955c2675e106ce55698bd401df47b0041ce7f`. The example paths and options were checked against that source; training runs and checkpoint accuracy were not independently reproduced for this documentation.

## Citation and license

When referencing this implementation, cite the repository URL and the exact commit used. If associated publication details are supplied, cite that publication separately; source-code target tables are not a substitute for a verified bibliographic reference.

Maintainer: [huzuarif2234](https://github.com/huzuarif2234).

No license file is present in the inspected revision. Contact the maintainer for reuse and redistribution terms.
