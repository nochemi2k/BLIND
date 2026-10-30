# BLIND

**Bimodal Learning from Imperfect NMR Data** - a chemistry-ignorant transformer model for autonomous molecular structure elucidation from experimental ¹H and/or ¹³C NMR spectra.

> **Publication:** Zlobin, I.; Bespalov, I.; Shandybo, M.; Rekut, N.; Nelyubina, Y.; Chusov, D.; Afanasyev, O.; Novikov, V. *Bimodal Learning from Imperfect NMR Data (BLIND): A chemistry-ignorant model for autonomous structure elucidation from NMR.* ChemRxiv, 2026. [https://doi.org/10.26434/chemrxiv.15006080.v1](https://doi.org/10.26434/chemrxiv.15006080.v1)

## Overview

BLIND translates routine 1D NMR peak lists - chemical shifts, integrals (for ¹H), and multiplicity labels - into molecular structures represented as SMILES strings. The model:

- Requires **no chemical priors** (no molecular formula, fragment libraries, or elemental composition)
- Is trained on **experimental spectra** from the chemical literature (>5.5 million spectra; see paper for dataset details)
- Handles **imperfect, heterogeneous** real-world NMR reporting formats
- Supports **unimodal** (¹H-only or ¹³C-only) and **bimodal** inference
- Predicts structures including **stereochemistry** via autoregressive SMILES generation

On the main test set, BLIND recovers the exact structure (including stereochemistry) in more than half of diverse cases; on an expert-curated external validation set, accuracy exceeds 80%. Full benchmarks and methodological details are in the [paper](https://doi.org/10.26434/chemrxiv.15006080.v1). The experimental-only large-scale dataset was acquired from [OdanChem database](https://odanchem.org/).

## Model

BLIND is a bimodal encoder–decoder Transformer. Each NMR peak is encoded as `(chemical shift, intensity, multiplicity)` using Fourier embeddings for continuous values and a learned vocabulary for multiplicity labels. Separate 6-layer encoders process ¹H and ¹³C spectra; a shared decoder with dual cross-attention autoregressively generates a SMILES string. During training, one modality is randomly dropped in 20% of bimodal examples to improve robustness to partial spectra.

| Parameter | Value |
|-----------|-------|
| `d_model` | 288 (shift 104, intensity 104, multiplicity 48, J couplings 32) |
| Encoder/decoder layers | 6 each, 8 heads |
| Optimizer | AdamW, lr = 3×10⁻⁴ |
| Batch size / epochs | 128 / 200 |
| Solvent filter | CDCl₃ |
| Top-N sampling defaults | `top_k=16`, `top_p=0.908`, `temperature=1.014` (from `models/optimized_params.json`) |

## Repository Structure

```
BLIND/
├── train.py, evaluate.py, evaluate_top_n.py   # Entry points
├── requirements.txt                           # Project dependencies
│
├── src/blind/
│   ├── training.py, evaluation.py, evaluation_top_n.py
│   ├── data_io.py                             # JSONL + CSV loading
│   ├── device.py                              # CUDA / MPS / CPU selection
│   ├── models/transformer.py
│   └── data/multiplicity_codes.json
│
├── models/
│   ├── smiles_tokenizer.model                 # SentencePiece tokenizer
│   └── optimized_params.json
│
├── checkpoints/
│   └── blind_trained_weights.pt               # Pre-trained weights
│
└── data/
    ├── samples/                               # Small test set (8 compounds)
    │   ├── train.jsonl, val.jsonl, test.jsonl
    └── external_validation_results.csv        # Full external validation set
```

See the paper for full data preparation details.

## Installation

**Python 3.10–3.12** recommended.

### CPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Apple Silicon (MPS)

```bash
pip install torch
pip install -r requirements.txt
```

### NVIDIA CUDA

Install a PyTorch build matching your GPU and driver from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```



## Data Format

### JSONL (training and evaluation)

Each line is one spectrum record:

```json
{
  "smiles": "CC(=O)NC(C)c1ccccc1",
  "spectrum_type": "H_NMR",
  "solvent": "CDCl_3",
  "spectrum": [
    {"signal": 7.29, "intensity": 5.0, "multiplicity": "m"},
    {"signal": 6.02, "intensity": 1.0, "multiplicity": "br_s"}
  ]
}
```

Multiple lines with the same `smiles` but different `spectrum_type` represent bimodal data for one molecule.

### External validation CSV

`data/external_validation_results.csv` can be passed directly to evaluation scripts. The `spectra_type` column (`H_only`, `C_only`, `both`) controls which spectra are used per compound.

| Column | Description |
|--------|-------------|
| `true_smiles` | Ground-truth structure |
| `H_spectrum` / `C_spectrum` | Input peak lists |
| `spectra_type` | Evaluation modality |
| `prediction_canonical` | Previously reported BLIND prediction |
| `H_doi` / `C_doi` | Source publication DOI |

## Training on NMRexp

`prepare_nmrexp.py` converts the [NMRexp](https://zenodo.org/records/17296666) parquet into `data/nmrexp/processed/` (70/20/10 splits by molecule and a new tokenizer). Each peak also carries `couplings`, the J values in Hz; the model uses the 4 largest per peak. Multiplicities use `src/blind/data/multiplicity_codes_nmrexp.json`; labels not listed there become `m`.

The peak embedding includes J constants, so `checkpoints/blind_trained_weights.pt` does not load with this model.

## Usage

> A user-friendly web interface (GUI) for this model is also available at [OdanChem](https://odanchem.org/predict-multimodal-compound-search), requiring no local installation.

### Training

```bash
# Default: auto device, sample data
python train.py

# NVIDIA GPU with mixed precision
python train.py --device cuda --mixed-precision

# Apple Silicon
python train.py --device mps
```

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--train-path` | `data/samples/train.jsonl` | Training data |
| `--val-path` | `data/samples/val.jsonl` | Validation data |
| `--tokenizer-path` | `models/smiles_tokenizer.model` | SentencePiece tokenizer |
| `--checkpoint-dir` | `checkpoints` | Checkpoint output directory |
| `--exp-prefix` | `blind_256` | Checkpoint filename prefix |
| `--batch-size` | 128 | Training batch size |
| `--epochs` | 200 | Number of epochs |
| `--num-workers` | `0` on Windows, `4` elsewhere | DataLoader workers |
| `--mixed-precision` | off | CUDA-only AMP |

### Evaluation (greedy decoding)

```bash
python evaluate.py \
    --checkpoint checkpoints/blind_trained_weights.pt \
    --data-path data/samples/test.jsonl \
    --output-json results.json

# External validation (CSV)
python evaluate.py \
    --checkpoint checkpoints/blind_trained_weights.pt \
    --data-path data/external_validation_results.csv \
    --output-json external_val_results.json
```

Reports exact-match accuracy and SMILES validity by modality: **Only C**, **Only H**, **Both**. Default `--max-tokens` is 275 (paper setting).

### Evaluation (top-N sampling)

```bash
python evaluate_top_n.py \
    --checkpoint checkpoints/blind_trained_weights.pt \
    --data-path data/external_validation_results.csv \
    --num-samples 10 \
    --output-json external_val_topn.json
```

Uses optimized sampling defaults from `models/optimized_params.json` (`top_k=16`, `top_p≈0.908`, `temperature≈1.014`; override with `--top-k`, `--top-p`, `--temperature`). Default `--num-samples` is 10 (top-10 evaluation). Reports top-10 exact-match accuracy and validity.

## Citation

```bibtex
@article{zlobin2026blind,
  title   = {Bimodal Learning from Imperfect {NMR} Data ({BLIND}): A chemistry-ignorant model for autonomous structure elucidation from {NMR}},
  author  = {Zlobin, Ivan and Bespalov, Ivan and Shandybo, Maksim and Rekut, Nikolai and Nelyubina, Yulia and Chusov, Denis and Afanasyev, Oleg and Novikov, Valentin},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15006080.v1},
  url     = {https://doi.org/10.26434/chemrxiv.15006080.v1}
}
```

## License

Licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). **Commercial use is not permitted.** See [LICENSE](LICENSE).

## Authors

- Ivan Zlobin, Ivan Bespalov, Maksim Shandybo - Kurnakov Institute of General and Inorganic Chemistry, RAS
- Nikolai Rekut, Denis Chusov, Oleg Afanasyev - Nesmeyanov Institute of Organoelement Compounds, RAS
- Yulia Nelyubina - Federal Research Center of Problems of Chemical Physics and Medicinal Chemistry, RAS
- Valentin Novikov - Universitat de Barcelona

Correspondence: novikov@ub.edu
