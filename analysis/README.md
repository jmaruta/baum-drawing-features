# analysis/

Scripts that reproduce the analyses in the paper. They expect the study tables described below, which contain participant-level data and are available from the corresponding author on reasonable request; they are **not** included here.

| Script | Input | Output |
|---|---|---|
| `kappa_cv.py` | `merged_features.csv` (features + human coding, column `site` = acquisition group) | `agreement_cv.csv`: nested cross-validated cut-offs, AUC, κ (bootstrap CI), sensitivity, specificity, leave-one-group-out κ |
| `perturb.py` → `perturb_agree.py` | folder of drawings | per-feature ICC(2,1) between original and perturbed copies (`perturb_icc_matrix.csv`) |
| `analysis.py` | `feats.csv`, `manual.csv`, `beh.csv` | associations with age, education, sex and psychological measures (`assoc.csv`), cross-validated prediction |
| `splithalf.py` | `merged_features.csv` | split-half reproducibility of associations |

Run each script with `python <script> --help`-style usage shown in its docstring.
