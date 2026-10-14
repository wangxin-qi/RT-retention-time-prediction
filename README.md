# RouteRT: retention-time prediction across LC methods

Source code for retention time prediction of liquid chromatography.

RouteRT combines molecular structure and LC-condition information to provide a global retention-time (RT) prediction. Depending on eligible evidence, the runtime can apply method-history correction, cross-condition RT transfer, method-family information, or post hoc calibration from a small target-method RT set. The submitted manuscript describes the evaluation and evidence-isolation protocol.

## What is in this repository

- `predict.py`: command-line launcher for the checked RouteRT prediction runtime.
- `train.py`: entry point for the global-backbone training module in `runtime/core/train/`.
- `runtime/core/`: model, training, prediction, routing, calibration, and LC-condition modules taken from the checked submission code bundle.
- `data/sample.csv`: three illustrative SMILES inputs. They are not rice-sample measurements and have no measured RT labels.
- `requirements.txt`: dependency names recorded in the submission code bundle.

The source modules and the public example are provided for code inspection. They are **not yet a standalone end-to-end reproduction package**: the processed training corpus, LC-method metadata, pretrained route weights, and full validation materials are not hosted here. Those materials are supplied separately in the manuscript's Supporting Information for Review Only archive. Running `train.py` requires the processed corpus and configured training paths; running a prediction requires the associated model assets and method metadata.

## Environment

Install the recorded dependencies in a suitable Python environment:

```bash
python -m pip install -r requirements.txt
```

The checked source requirements file did not pin versions. Exact tested versions and an executable public release protocol remain to be documented; do not assume that installing the latest packages reproduces the manuscript results.

## Example input

`data/sample.csv` contains the columns `compound_id` and `smiles`. It is a format example only. Prediction additionally requires a matching LC-method description and the trained model assets. No predicted values are supplied for this illustrative file.

## Data, weights, and reuse

Large raw LC–MS files, third-party datasets, and pretrained checkpoint collections are excluded from this public repository. The authors will review redistribution rights, public hosting, and a versioned release before publication. No software license has yet been selected; repository visibility alone does not grant reuse rights.

## Citation

A formal citation and versioned release identifier will be added when available. Until then, cite the RouteRT manuscript if it is appropriate and available.
