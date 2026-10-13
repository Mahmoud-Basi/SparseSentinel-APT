# SparseSentinel-APT

Official research code and private data package for **Multimodal Aspect-Based Threat-Sentiment Analysis for APT Detection**.

SparseSentinel-APT formulates APT investigation as multimodal aspect-based threat-sentiment analysis (MABTSA). It combines aspect-conditioned multimodal joint-embedding predictive learning (AM-JEPA), sparse supporting/contradicting evidence selection, uncertainty-aware abstention, and deterministic reports linked to immutable evidence identifiers.

## Release status

This repository is being kept **private** while the manuscript is under development. The included 240-unit stance pilot is an **AI-assisted single-reviewer silver-label resource**, not an independently adjudicated human gold standard. Do not make deployment or human-level-performance claims from it.

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
| Dense AM-JEPA stance macro-F1, silver pilot | 0.5680 |
| Text sparse stance macro-F1, silver pilot | 0.5670 ± 0.0398 |
| Text sparse selective macro-F1, silver pilot | 0.6246 at 69.2% coverage |
| GUIDE dense-attention stance macro-F1, 263,840 test units | 0.7906 ± 0.0047 |
| GUIDE single-sparse stance macro-F1, K=4 | 0.7912 ± 0.0031 |
| GUIDE dual-sparse stance macro-F1, K=4 | 0.7870 ± 0.0080 |
| GUIDE dense-attention selective macro-F1 | 0.8469 at 81.7% test coverage |
| GUIDE dense-attention macro-F1, unseen organizations | 0.7486 ± 0.0028 |

## Repository layout

```text
notebooks/                    End-to-end research notebook
SparseSentinel-GUIDE.ipynb    GUIDE data audit and stance-unit construction
src/                          Experiment stages and annotation-package construction
data/                         Silver pilot, taxonomy/guidelines, and release notices
figures/                      Publication figures
docs/                         Audit and methodology notes
configs/                       Reproducibility configuration
```

## Installation

```bash
conda env create -f environment.yml
conda activate sparsesentinel-apt
jupyter lab
```

Alternatively:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## Reproduction order

Open `notebooks/SparseSentinel_APT_Final_Notebook.ipynb`. The extracted scripts in `src/experiments/` correspond to the stance and final evaluation stages:

1. `cell_26_stance_experiment.py`
2. `cell_27_sparse_selector.py`
3. `cell_28_dual_selector.py`
4. `cell_29_stance_uncertainty.py`
5. `cell_30_final_ablation_robustness.py`
6. `cell_31_statistics_integrity_figures_exports.py`

The separate `SparseSentinel-GUIDE.ipynb` notebook is an initial GUIDE evaluation workflow. G0 checks the environment, G1 audits the downloaded CSV files and converts them to Parquet, and G2 constructs incident-by-aspect stance units and evidence windows. The notebook currently marks the probe, sparse-selection, and abstention stage (G3) as a next step; it does not yet report GUIDE model-performance results. It expects `GUIDE_Train.csv` and `GUIDE_Test.csv` in `GUIDE/` at the repository root. See the notebook before running it.

All preprocessing and model-selection operations must be fitted on training/validation partitions only. The held-out test labels must not be accessed during configuration selection.

## Data availability

Raw DAPT-2020 traffic is **not redistributed**. Obtain it from its original provider and comply with its terms. This private package includes only the derived silver-label pilot and metadata required for internal manuscript reproduction. See `data/README.md` and `data/DATA_USE_NOTICE.md` before using or sharing it.

GUIDE data are also not included. Download the Microsoft Security Incident Prediction dataset from [Kaggle](https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction) and place its `GUIDE_Train.csv` and `GUIDE_Test.csv` files in `GUIDE/` before running the GUIDE notebook. The notebook treats `IncidentGrade` as the analyst-assigned label and excludes `ActionGrouped` and `ActionGranular` from model features. Check the dataset license and provider terms before use or redistribution.

The following are deliberately excluded:

- raw DAPT-2020 and CIC-IDS2017 files;
- hidden provenance/linkage keys;
- individual annotator and adjudicator workbooks;
- private prompts and agent intermediate files;
- tensors, embeddings, checkpoints, caches, and API responses.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated paper after publication.

## License

Source code is released under the MIT License. The derived data are governed separately by `data/DATA_USE_NOTICE.md` and remain subject to the upstream DAPT-2020 terms.

