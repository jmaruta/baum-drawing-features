# baum-drawing-features

Rule-based image analysis of tree drawings (Baum test): size-invariant features of **line quality** and **structure** extracted from ordinary scanned drawings. No training data, no special hardware.

Companion code for:

> Maruta J, Taniguchi M, Sugimoto N, Horimoto M, Kurozumi H, Inoue K. *Rule-based image analysis of tree drawings in older adults: agreement with human coding, reliability, and correlates.* (submitted)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](10.5281/zenodo.24182820 ) 

## What it does

`features.py` takes a scanned drawing (PNG/JPG/TIFF), flattens the scanner background, binarises the drawing with a contrast-adaptive hysteresis threshold, removes paper edges and printed text, crops the tree and rescales it, and computes 27 features that do not depend on the absolute size of the tree:

| Domain | Features |
|---|---|
| Line quality | line darkness (pen-pressure proxy), stroke width and its variability, shading / filled area, line jaggedness, high-frequency turning, line gaps, short-segment fraction, small fragments, number of components, darkness variability |
| Structure | total line length, ink coverage, end-point density, junction density (branching), small loops (leaves / fruit), four crown-closure measures, trunk line count (single / double line), crown-to-trunk width ratio, ground-line extent, left–right asymmetry, ink in upper half, vertical centroid, height / width ratio |

Processing takes about 0.1 s per drawing on one CPU core. Exact definitions are given in Table 2 of the paper and in the docstrings of `features.py`.

**What it does not do.** It does not diagnose anything, and it does not reproduce every qualitative sign used by human raters: in our data, double-line trunk, crown closure and ground line were reproduced with moderate agreement (κ 0.44–0.52), whereas leaves and fruit were not captured. Line-quality features depend on scan resolution and contrast; see *Recommendations* below.

## Quick start

### Option A — in the browser (no installation)

Open `notebooks/baum_features_colab.ipynb` in Google Colab, upload your scanned drawings, run all cells, and download the resulting CSV together with the binarised images for checking.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jmaruta/baum-drawing-features/blob/main/notebooks/baum_features_colab.ipynb) 

### Option B — on your computer

```bash
git clone https://github.com/jmaruta/baum-drawing-features.git
cd baum-drawing-features
pip install -r requirements.txt
python features.py examples features_examples.csv debug_images
```

`features_examples.csv` holds one row per drawing (column `ID` = file name); `debug_images/` holds the binarised drawing of every image. **Always look at the debug images**: faint pencil lines and printed text near the tree are the two common failure modes.

To analyse your own drawings, point the script at a folder of image files (or a folder of sub-folders):

```bash
python features.py /path/to/scans my_features.csv my_debug
```

## Recommendations for acquisition

* Scan so that the tree spans **at least ~600 px** on its long side (≈150 dpi for a tree occupying a third of an A4 sheet). Structural features are robust down to about half that; the fine line-quality features (jaggedness, high-frequency turning, stroke width) are not.
* Keep scanner, resolution and pen type **constant within a study**. Line darkness and stroke width are comparable only among drawings acquired under the same conditions.
* Prefer `jagged` over `turn_hf` as the measure of line irregularity (more robust to resolution and more stable over time).
* `ink_dark_cv` (darkness variability) was not robust in our tests and is reported for completeness only.
* Blank sheets allow size and placement on the page to be measured as well (`page_*` columns in the paper); drawings made in a questionnaire margin do not.

## Repository layout

```
features.py                   feature extraction (the only file needed to use the method)
requirements.txt              Python dependencies
notebooks/baum_features_colab.ipynb   browser-based runner
examples/                     three synthetic drawings for testing (no participant data) and the script that made them
analysis/
  kappa_cv.py                 agreement with human coding: nested cross-validated cut-offs, κ, sensitivity, specificity
  perturb.py, perturb_agree.py  measurement robustness: perturbed copies -> ICC(2,1)
  analysis.py, splithalf.py   associations with participant characteristics and split-half reproducibility
CITATION.cff                  citation metadata
LICENSE                       MIT
```

The scripts in `analysis/` reproduce the analyses of the paper but require the study tables (features merged with human coding and questionnaire data), which are available from the corresponding author on reasonable request. The drawings themselves cannot be shared.

## Data and ethics

No participant drawings or individual data are included in this repository. The example images in `examples/` are synthetic. The analyses reported in the paper were approved by the Ethics Committee of Osaka Metropolitan University (approvals 2023-103 and 2025-108).

## Citation

If you use this code, please cite the paper above and the Zenodo archive of the release you used (see `CITATION.cff`).

## License

MIT License — see `LICENSE`.
