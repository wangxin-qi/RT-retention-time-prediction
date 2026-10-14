"""Entry point for the RouteRT global-backbone training code.

Training requires the processed corpus, LC-method metadata, and route assets
described in the manuscript. Those files are not part of this public repository.
The complete checked review bundle is supplied separately with the submission.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime"))

from core.train.train_v11_pro_meta_threehead_v47_calibrated_gate import main


if __name__ == "__main__":
    main()
