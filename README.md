# SparseSentinel-APT

Research code and derived data package for **Multimodal Aspect-Based Threat-Sentiment Analysis for APT Detection**.

SparseSentinel-APT formulates APT investigation as multimodal aspect-based threat-sentiment analysis (MABTSA). It combines aspect-conditioned multimodal joint-embedding predictive learning (AM-JEPA), sparse supporting/contradicting evidence selection, uncertainty-aware abstention, and deterministic reports linked to immutable evidence identifiers.

## Release status

> **Anonymized for double-blind review.** This repository is shared through an anonymous link solely for peer review of the associated manuscript. Author names, affiliations, and contact details have been removed. Please do not attempt to identify the authors. A public, de-anonymized release will follow after the review process.

The included 240-unit stance pilot is an **AI-assisted single-reviewer silver-label resource**, not an independently adjudicated human gold standard. Do not make deployment or human-level-performance claims based on it.

The pipeline does not use a post-hoc LLM verifier. Its terminal action is either:

1. a deterministic evidence-linked report, or
2. abstention under validation-selected confidence/evidence rules.

## Main results

| Evaluation | Result |
|---|---:|
| Campaign F1, five seeds | 0.8699 ± 0.0043 |
| Campaign AUROC, five seeds | 0.9469 ± 0.0063 |
| Core-three macro-F1, five seeds | 0.8381 ± 0.0157 |
| Missing-network campaign F1 | 0.8143 ± 0.0226 |
| Missing-text campaign F1 | 0.8343 ± 0.0202 |
| Dense AM-JEPA stance macro-F1 (silver pilot) | 0.5680 |
| Text sparse model, full-coverage macro-F1 (silver pilot) | 0.5670 ± 0.0398 |
| Text sparse model, selective macro-F1 (silver pilot) | 0.6246 at 69.2% coverage |
| GUIDE dense-attention stance macro-F1 (263,840 test units) | 0.7906 ± 0.0047 |
| GUIDE single-sparse stance macro-F1 ($K=4$) | 0.7912 ± 0.0031 |
| GUIDE dual-sparse stance macro-F1 ($K=4$) | 0.7870 ± 0.0080 |
| GUIDE dense-attention selective macro-F1 | 0.8469 at 81.7% test coverage |
| GUIDE dense-attention macro-F1, unseen organizations | 0.7486 ± 0.0028 |

Silver-pilot results are reported on AI-assisted silver labels and should be interpreted as indicative only.

## Repository layout

```text
notebooks/                    End-to-end research notebook
SparseSentinel-GUIDE.ipynb    GUIDE data audit and stance-unit construction
src/                          Experiment stages and annotation-package construction
data/manifests/               Dataset manifests and annotation metadata
data/silver_pilot/            Derived DAPT2026-MABTSA silver-pilot units
data/taxonomy/                Aspect taxonomy and annotation guidance
figures/                      Publication figures
docs/                         Audit and methodology notes
configs/                      Reproducibility configuration
```

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate sparsesentinel-apt
jupyter lab
```

Alternatively, using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab
```

## Reproduction order

### Main pipeline

Open `notebooks/SparseSentinel_APT_Final_Notebook.ipynb`. The extracted scripts in `src/experiments/` correspond to the stance and final evaluation stages and should be run in this order:

1. `cell_26_stance_experiment.py`
2. `cell_27_sparse_selector.py`
3. `cell_28_dual_selector.py`
4. `cell_29_stance_uncertainty.py`
5. `cell_30_final_ablation_robustness.py`
6. `cell_31_statistics_integrity_figures_exports.py`

### GUIDE workflow

`SparseSentinel-GUIDE.ipynb` contains the initial GUIDE data workflow:

- **G0** checks the environment.
- **G1** audits the CSV files and converts them to Parquet.
- **G2** constructs incident-by-aspect stance units and evidence windows.

The GUIDE performance results in the table above are reported in Experiment 6 of the manuscript. The current notebook stops before the probe, sparse-selection, and abstention evaluation stage (G3), so it does **not** by itself reproduce those results. The notebook expects `GUIDE_Train.csv` and `GUIDE_Test.csv` in a `GUIDE/` folder at the repository root.

### Evaluation protocol

All preprocessing and model-selection operations are fitted on training/validation partitions only. Held-out test labels must not be accessed during configuration selection.

## Data availability

### DAPT2026-MABTSA (derived)

The repository includes the derived DAPT2026-MABTSA silver-pilot data under `data/silver_pilot/`, with manifests and annotation metadata under `data/manifests/` and taxonomy/guidance under `data/taxonomy/`. Raw DAPT-2020 traffic is **not redistributed**; obtain it from its original provider and comply with its terms. Read `data/README.md` and `data/DATA_USE_NOTICE.md` before using or sharing the included derived data.

### GUIDE

GUIDE data are not included. Download the Microsoft Security Incident Prediction dataset from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) and place `GUIDE_Train.csv` and `GUIDE_Test.csv` in `GUIDE/` before running the GUIDE notebook. The notebook treats `IncidentGrade` as the analyst-assigned label and excludes `ActionGrouped` and `ActionGranular` from model features. Check the dataset license and provider terms before use or redistribution.

### Deliberately excluded

- raw DAPT-2020 and CIC-IDS2017 files;
- hidden provenance/linkage keys;
- individual annotator and adjudicator workbooks;
- private prompts and agent intermediate files;
- tensors, embeddings, checkpoints, caches, and API responses.

## Citation

Citation information will be provided after the review process concludes.

## License

Source code is released under the MIT License (see `LICENSE`). The derived data are governed separately by `data/DATA_USE_NOTICE.md` and remain subject to the upstream DAPT-2020 terms.
