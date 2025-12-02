# Sequence-context-aware decoding enables quantitative recovery of protein dynamics from crystallographic B-factors

## Overview

X-ray crystallography provides the majority of protein structures, yet the B-factors associated with these coordinates are often influenced by crystal packing and refinement protocols, limiting their utility for quantifying solution-state dynamics. Consequently, a gap remains between the abundance of static PDB structures and the availability of quantitative dynamic information. 

Here, we investigate the extent to which sequence context can help recover intrinsic motion from crystallographic data. We present the **B-Factor Corrector (BFC)**, a fine-tuned protein language model that treats B-factor analysis as a sequence-to-dynamics translation task. By leveraging deep contextual embeddings, BFC separates crystal lattice effects from intrinsic flexibility, achieving a Pearson correlation of **0.80** with ground-truth molecular dynamics simulations. Furthermore, the model improves the estimation of absolute fluctuations (in Å, *r*=0.49), suggesting that large-scale static structural data can be effectively repurposed to inform quantitative dynamic ensembles.

## Google Colab Notebooks

**Status**: Under development [2025-12-03]

Interactive Google Colab notebooks for training, prediction, and visualization will be available soon. Stay tuned!

## Repository Structure

```
.
├── README.md                 # This file
├── bfc_model/               # Pre-trained BFC models
│   ├── bfc_model_esm_finetuned.pth
│   ├── bfc_model_esm_frozen_linear.pth
│   └── bfc_model_xgboost_bfactor_only.json
├── scripts/                 # Python scripts for training, prediction, and analysis
│   ├── train.py            # Model training script
│   ├── predict_single_pdb.py  # Single PDB prediction
│   ├── prepare_data.py     # Data preprocessing
│   ├── ablation.py         # Ablation study experiments
│   ├── figure1.py          # Generate Figure 1
│   ├── figure2.py          # Generate Figure 2
│   ├── supp2.py            # Supplementary analysis
│   ├── run_xgboost.py      # XGBoost baseline
│   ├── analyze_length.py   # Sequence length analysis
│   ├── get_pdb.py          # PDB file retrieval
│   ├── pachong.py          # Web scraping utilities
│   ├── rmsf.py             # RMSF calculation
│   ├── test.py             # Testing utilities
│   ├── debug.py            # Debugging utilities
│   └── ATLAS/              # ATLAS dataset validation experiments
│       ├── infer.py        # Inference script for ATLAS data
│       ├── plot.py         # Plotting utilities
│       ├── plot2.py        # Additional plotting utilities
│       ├── *.pdb           # PDB structure files
│       ├── *_Bfactor.tsv   # B-factor data
│       ├── *_RMSF.tsv      # RMSF ground truth data
│       └── *_prediction_normalized.csv  # Prediction results
├── data/                    # Data files
│   ├── all_member_ids.txt  # Dataset member IDs
│   └── clusterInfoTable.txt # Cluster information
└── notebooks/               # Jupyter notebooks (under development)
```
