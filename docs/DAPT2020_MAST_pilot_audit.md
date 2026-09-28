# DAPT2020-MAST Pilot Annotation Audit

Audit date: 2026-09-23  
Pilot size: 240 annotation units  
Evidence per unit: 32 ordered events

## Integrity checks

| Check | Result |
|---|---:|
| Annotator A records | 240 |
| Annotator B records | 240 |
| IDs matched across passes | 240 |
| Duplicate IDs | 0 |
| Missing required annotation fields | 0 |
| Invalid stance, context, or confidence values | 0 |
| Records flagged for resolution | 81 |
| Completed resolutions | 81 |

## Pass-to-pass diagnostic agreement

| Field | Agreements | Rate |
|---|---:|---:|
| Stance | 213/240 | 88.75% |
| Insufficient context | 186/240 | 77.50% |
| Exact agreement across the compared annotation fields | 159/240 | 66.25% |

The 81 flagged records comprise 27 stance disagreements and 54 insufficient-context disagreements. Confidence had a quadratic-weighted pass-to-pass kappa of 0.794 and mean absolute difference of 0.208, but these values are workflow diagnostics only.

## Resolved pilot distribution

| Final field | Count |
|---|---:|
| Supporting | 56 |
| Contradicting | 38 |
| Neutral | 146 |
| Insufficient context = TRUE | 32 |
| Insufficient context = FALSE | 208 |

| Aspect | Units | Supporting | Contradicting | Neutral |
|---|---:|---:|---:|---:|
| Data exfiltration | 50 | 17 | 0 | 33 |
| Lateral movement | 60 | 14 | 13 | 33 |
| Initial access / foothold | 59 | 1 | 0 | 58 |
| Reconnaissance | 71 | 24 | 25 | 22 |

## Provenance and publication status

The two initial passes were generated and were not independent human expert annotations. The 81 flagged records were resolved through an evidence-based single-reviewer pass. Accordingly, this pilot must be described as an **AI-assisted, single-reviewer-resolved silver-label resource**. Its agreement values must not be presented as human inter-rater reliability, and the labels must not be called an expert gold standard.

The pilot is suitable for annotation-protocol validation, pipeline development, and explicitly qualified exploratory stance experiments. Independent human validation remains necessary before confirmatory stance-performance or gold-standard benchmark claims.

