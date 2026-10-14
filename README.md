# RT-retention-time-prediction

Source code for retention time prediction of liquid chromatography.

This repository is being prepared as the public code companion to the RouteRT manuscript. RouteRT combines molecular structure and LC conditions for a global retention-time (RT) prediction and can use eligible method-history, cross-condition RT, or related-method evidence for correction or transfer. A small set of measured RTs from a target method can support post hoc calibration.

## Repository status

The public repository is **under construction**. The training script, prediction script, a small example dataset, and any distributable model checkpoint will be added after source-code and data-sharing review. The files currently present do **not** constitute a complete reproducibility release. The manuscript's review-only materials are supplied separately through the journal submission system.

## Planned layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                 # pending
├── predict.py               # pending
├── data/
│   └── sample.csv           # pending; small example only
└── model/
    └── checkpoint.pth       # optional; pending rights and size review
```

## Environment

A Python environment is required. Install the currently recorded project dependencies with:

```bash
python -m pip install -r requirements.txt
```

The dependency list is taken from the checked RouteRT code package, but its source file did not specify package versions. Tested versions and executable training/inference instructions will be documented when the public scripts are added.

## Data and model availability

Large raw chromatographic files and third-party datasets are not included in this repository. Only a minimal shareable example will be added here. Any pretrained checkpoint will be released only after confirming redistribution rights and an appropriate hosting method.

## Citation and contact

A citation and release identifier will be added when the manuscript and public repository are finalized. For questions, contact the corresponding authors identified in the manuscript.
