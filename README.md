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

### Heteronuclear spectra (19F, 31P, 11B, 29Si)

`prepare_nmrexp_hetero.py` copies `data/nmrexp/processed/` to `data/nmrexp/processed_hetero/` and adds the CDCl3 heteronuclear spectra (`F_NMR`, `P_NMR`, `B_NMR`, `Si_NMR` records) of the same molecules; splits and tokenizer are unchanged. The model (`src/blind/models/hetero_transformer.py`) adds a small encoder per nucleus and a cross-attention per nucleus in every decoder layer. `hetero/` runs the main training and evaluation code with it, without changing the main code.

```bash
python prepare_nmrexp_hetero.py

# Fine-tune a trained 1H/13C model; each epoch: all molecules with heteronuclear spectra + 3 times as many without
python hetero/train.py \
    --init-checkpoint checkpoints/<1H/13C model>.pt \
    --train-path data/nmrexp/processed_hetero/train.jsonl.gz \
    --val-path data/nmrexp/processed_hetero/val.jsonl.gz \
    --tokenizer-path data/nmrexp/processed/smiles_tokenizer.model \
    --lr 1e-4 --epochs 30 --exp-prefix hetero

# Evaluate; --no-hetero hides the heteronuclear spectra (same weights, ablation)
python hetero/evaluate.py \
    --checkpoint checkpoints/hetero_epoch_29.pt \
    --data-path data/nmrexp/processed_hetero/test.jsonl.gz \
    --tokenizer-path data/nmrexp/processed/smiles_tokenizer.model \
    --output-json results/hetero.json
```

The 1H/13C weights are loaded as they are; heteronuclear encoders and cross-attentions start from random weights with a zero output projection, so before fine-tuning the model reproduces the base model.

During fine-tuning each heteronuclear spectrum is dropped with probability 0.2; the 13C/1H dropout is unchanged, so at least one spectrum always remains. A nucleus absent from the molecule (no F, P, B or Si atom) is given as an empty spectrum with probability 0.5: a learned vector per nucleus that the decoder attends to, unlike a spectrum that is not given at all. A nucleus present in the molecule without a spectrum is never given as empty. In evaluation, `--empty-hetero` gives the empty spectrum for every absent nucleus (taken from the true structure, as if the spectrum were measured and showed no signals).

### Elemental analysis noise model

There is no public dataset of elemental analyses with structures, so the analysis is computed from the SMILES and made realistic with noise taken from published data. `prepare_elemental_analysis.py` downloads open-access PubMed Central articles that report "Anal. Calcd for ...: C, ...; H, .... Found: ..." (Europe PMC search, NCBI E-utilities full text), keeps the statements whose calculated values match the formula and writes `data/ea/ea_noise.json`: how often each element is reported when present, the neutral solvates added to formulas, and the Found − Calcd vectors.

```bash
python prepare_elemental_analysis.py --num-articles 5000   # ~45 min; --skip-download to only re-parse
```

`src/blind/elemental_analysis.py` draws a "Found" analysis for a molecule: the reported elements, a solvate with the observed frequency, and a Found − Calcd vector resampled from analyses that report the same elements, so the spread, tails and the share of values fitted to the calculated ones follow real data.

```python
import numpy as np
from blind.elemental_analysis import ElementalAnalysisNoise, formula_counts
noise = ElementalAnalysisNoise("data/ea/ea_noise.json")
noise.sample(formula_counts("CC(=O)NC(C)c1ccccc1"), np.random.default_rng(0))  # {'C': ..., 'H': ..., 'N': ...}
```

### Experimental elemental analyses with structures

`ea/build_dataset.py` turns the downloaded articles into a dataset of experimental analyses: SMILES, molecular formula, Calcd and Found values, source article. The structure comes from the name of the compound in the text before the analysis and is kept only if its composition equals the formula of the analysis. Names are converted by [OPSIN](https://github.com/dan2097/opsin) (systematic names; needs Java, e.g. `conda install -c conda-forge openjdk`, and the OPSIN jar) and by `ea/ligands.py` for complexes (bracket formulas such as `[Cu(phen)2Cl]ClO4`, additive names, and ligands labelled in the same article, e.g. `[Os(p-cym)Cl(L1)]Cl` once L1 is resolved). Complexes are written as disconnected ions.

```bash
python ea/build_dataset.py --input-dirs data/ea data/ea/metals data/ea/complexes data/ea/general \
    --opsin-jar data/ea/tools/opsin.jar --output-dir data/ea/dataset
```

Output: `ea_dataset.jsonl.gz` (all analyses, `smiles` null when the name was not resolved), `ea_dataset_with_smiles.csv` and `summary.json`.

Statements are found by `src/blind/ea_statements.py`: two adjacent blocks of element-value pairs (or one block of "value (value)" pairs) next to the words calcd/found/requires, in any order and style ("Anal. Calcd for X: ... Found: ...", "Found: ... X requires ...", "X (MW): calcd. ...; found ...", "found (calcd)"), with the formula taken from any formula in the text around them whose composition gives the calculated values (it may also be only in the HRMS line). Names are tried as written and in cleaned forms (trailing labels and section numbers removed, "Hydrochloride of X" reordered, words joined); a compound named only by its label (11, 3a, L1) gets the names written before "(11)" elsewhere in the article (`smiles_source` = `opsin_label`); a SMILES that differs from the formula by HCl, HBr, HI, H2O, TFA or MsOH gets those fragments added or removed.

Articles and supporting information can also be downloaded with `bash ea/run_download.sh start` (checks, a test download, then a background run of `ea/download_corpus.py`; `status`, `stop`, `log`).

The analyses left unresolved are also written to `llm_todo.jsonl` (candidate names + target formula without solvates) for translation by an LLM, mainly for complexes whose ligands are trivial names or abbreviations. `ea/llm_smiles.py split` cuts it into batches, the LLM writes `llm_batches/batch_XXXX.answers.jsonl` and checks them with `ea/llm_smiles.py check`, `ea/llm_smiles.py merge` collects the formula-verified SMILES into `llm_smiles.json`, and the next `build_dataset.py` run uses them (`smiles_source` = `llm`). A matching formula does not prove the structure (isomers, ligands guessed from the formula), so these rows are less reliable than OPSIN ones and can be filtered by `smiles_source`.

### Learned generator of elemental analyses

`src/blind/ea_noise_flow.py` is a generative model of Found values for C, H, N, S given only the molecular formula, trained by `ea/train_noise_flow.py` on the experimental dataset above (organic compounds and metal complexes together; no compound class is given). It models Δ = Found − Calcd, with Calcd computed from the SMILES plus the acids and counterions of the formula (HCl, TFA, PF6, ...) but not the neutral solvates: their shift of the Found values (a hydrate, residual CH2Cl2) is part of the learned noise, as a real sample holds them too. The density is autoregressive over C → H → N → S, each factor a mixture of 8 Gaussians from an MLP over formula features (mass % of the elements and of metals, weight, atom ratios: what an analysis tells anyway, no structural features, so the noise cannot hint at the structure), the previous deltas and their observed flags (an autoregressive flow: exact likelihood, sampling in one pass). Elements the article did not report are masked out of the likelihood, so sulfur (10k analyses) does not limit C, H, N (90k). Elements absent from the molecule but measured ("N, 0.00") take the empirical Found values of such reports. Analyses with |Δ| > 5 (wrong structure or formula) are left out; train / val / test are split by article.

```bash
python ea/train_noise_flow.py --dataset data/ea/dataset/ea_dataset.jsonl.gz --output data/ea/noise_flow.pt
```

`data/ea/noise_flow.report.json` compares it on the test articles with independent Gaussians per element and with the flow given only the calculated % of C, H, N, S (NLL), gives the coverage of its 50 % / 90 % intervals and the AUC of a classifier that tells real from generated analyses of the same molecules (0.5 = indistinguishable), also for the resampling model.

`--model spline` trains the same autoregressive model with neural spline flow factors instead of Gaussian mixtures (Durkan et al. 2019): each delta is mapped by asinh(Δ / 0.1), linear at the sharp peak near zero and logarithmic in the tails, then by a monotone rational-quadratic spline (16 bins) whose knots come from the formula, the previous deltas and their flags, to a standard normal. `--compare` adds another checkpoint to the report.

```bash
python ea/train_noise_flow.py --model spline --output data/ea/noise_flow_spline.pt --compare data/ea/noise_flow.pt
```

Both checkpoints load with `ElementalAnalysisFlow`. `sample(..., with_other=True)` adds Other = 100 − C − H − N − S, so the five parts sum to 100 %. Other is not modelled on its own: it is not measured, unknown whenever an element of the compound is not reported (a quarter of the analyses), and C + H + N + S exceed 100 % in about 1 % of real analyses of compounds made only of these elements, so a strict simplex (log-ratio) model would not fit the data. The report compares the share of such analyses and the spread of Other − Calcd, real and generated.

```python
from blind.ea_noise_flow import ElementalAnalysisFlow
flow = ElementalAnalysisFlow("data/ea/noise_flow.pt")
flow.sample(["CC(=O)NC(C)c1ccccc1"], num_samples=4)  # [1, 4, 4] Found % of C, H, N, S
```

### Elemental analysis as input

The model with elemental analysis (`src/blind/models/ea_transformer.py`) extends the model with heteronuclear spectra: a small encoder over the reported elements, one token per element (element embedding, Fourier features of the Found mass % and of the molar % among the reported elements, both in absolute %, the scale of the measurement error), and a cross-attention to it in every decoder layer. `ea/` runs the main training and evaluation code with it, on top of `hetero/`: the same data, sampling, heteronuclear dropout and empty spectra. The weights of a trained model with heteronuclear spectra are loaded as they are; the new parts start with a zero output projection, so before fine-tuning the model reproduces that model.

Analyses come from the learned generator (`EA_NOISE = "flow"` in `ea/enable_ea.py`; 16 per molecule generated when the dataset is built) and follow what a user enters: C and H always, N for 95 % of molecules with nitrogen, S for half of those with sulfur, and an element absent from the molecule is sometimes given as measured (≈ 0.00; N 50 %, S 25 %). A measured element is a token, even at 0.00; an element not measured has no token, so "measured, zero" and "unknown" are different inputs. `EA_NOISE = "resampling"` keeps the resampling model with all reported elements. Each time a molecule is seen in training one of its analyses is picked and the whole analysis is dropped with probability 0.3; in evaluation every example gets a fixed analysis seeded by its SMILES. `--no-ea` hides it; `--no-hetero` and `--empty-hetero` work as in `hetero/`.

```bash
python ea/train.py \
    --init-checkpoint checkpoints/<model with heteronuclear spectra>.pt \
    --train-path data/nmrexp/processed_hetero/train.jsonl.gz \
    --val-path data/nmrexp/processed_hetero/val.jsonl.gz \
    --tokenizer-path data/nmrexp/processed/smiles_tokenizer.model \
    --lr 1e-4 --epochs 30 --exp-prefix ea

python ea/evaluate.py \
    --checkpoint checkpoints/ea_epoch_29.pt \
    --data-path data/nmrexp/processed_hetero/test.jsonl.gz \
    --tokenizer-path data/nmrexp/processed/smiles_tokenizer.model \
    --output-json results/ea_29.json
```

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
