# run_xgboost.py

import os
import json
import argparse
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import time

# --- 1. Data Preparation: B-Factor Only ---
def prepare_data_from_bfactor(jsonl_file):
    """
    Loads data from a JSONL file and extracts only the B-factor as the
    feature (X) and RMSF as the label (y).

    Args:
        jsonl_file (str): Path to the .jsonl data file.

    Returns:
        X (np.array): Feature matrix of shape (total_residues, 1).
        y (np.array): Target vector of shape (total_residues,).
    """
    print(f"Loading data from {jsonl_file} (B-Factor Only)...")
    
    all_features = []
    all_labels = []

    with open(jsonl_file, 'r') as f:
        for line in tqdm(f, desc=f"Processing {os.path.basename(jsonl_file)}"):
            item = json.loads(line)
            
            # Feature: Normalized B-factor. Reshape to be (n_samples, 1).
            norm_b = np.array(item['Norm_B_factor']).reshape(-1, 1)
            all_features.append(norm_b)
            
            # Label: Normalized RMSF
            labels = np.array(item['Norm_RMSF'])
            all_labels.append(labels)

    # Concatenate all data into final X and y matrices
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"Data preparation complete. Feature matrix shape: {X.shape}, Label vector shape: {y.shape}")
    return X, y

# --- 2. Main Training and Saving Script ---
def main(args):
    """
    Main function to orchestrate data loading, model training, and saving.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🚀 STARTING XGBOOST BASELINE EXPERIMENT (B-Factor Only)")
    print("   Training and saving model without final evaluation.")
    print("="*80)

    # 1. Prepare data
    X_train, y_train = prepare_data_from_bfactor(args.train_file)
    X_val, y_val = prepare_data_from_bfactor(args.val_file)

    # 2. Define XGBoost model parameters
    base_params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',  # <<<--- FIX: Moved parameter here
        'random_state': 42,
        'early_stopping_rounds': 50,
        'n_jobs': args.n_jobs # Set number of CPU threads
    }

    # Configure for GPU or CPU based on device argument
    if "cuda" in args.device:
        print(f"Attempting to configure XGBoost for GPU on device: {args.device}")
        gpu_params = base_params.copy()
        gpu_params.update({'tree_method': 'gpu_hist', 'device': args.device})
        try:
            model = xgb.XGBRegressor(**gpu_params)
            print(f"Successfully initialized XGBoost with tree_method='gpu_hist' on {args.device}")
        except xgb.core.XGBoostError as e:
            print(f"Warning: GPU initialization failed with error: {e}")
            print("Falling back to CPU-based training ('hist').")
            cpu_params = base_params.copy()
            cpu_params.update({'tree_method': 'hist'})
            model = xgb.XGBRegressor(**cpu_params)
    else:
        print("Configuring XGBoost for CPU-based training.")
        cpu_params = base_params.copy()
        cpu_params.update({'tree_method': 'hist'})
        model = xgb.XGBRegressor(**cpu_params)

    # 3. Train the model
    # The model will use the validation set for early stopping to find the best iteration.
    print(f"\nTraining XGBoost model using {args.n_jobs} parallel processes...")
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        # eval_metric='rmse', # <<<--- FIX: Removed from here
        verbose=100
    )
    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")

    # 4. Save the trained model
    save_name = "bfc_model_xgboost_bfactor_only.json"
    save_path = os.path.join(args.output_dir, save_name)
    model.save_model(save_path)
    print(f"\n✅ Model saved to {save_path}")
    print("="*80 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and save a simple XGBoost model using only B-factors.")
    parser.add_argument("--train_file", type=str, default="processed_data/train.jsonl", help="Path to training data (.jsonl)")
    parser.add_argument("--val_file", type=str, default="processed_data/validation.jsonl", help="Path to validation data (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="bfc_model", help="Directory to save the trained model")
    
    # --- XGBoost Hyperparameters ---
    parser.add_argument("--n_estimators", type=int, default=1000, help="Number of boosting rounds (trees)")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate (eta) for XGBoost")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum depth of a tree")
    
    # --- Hardware Specific ---
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use for training (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--n_jobs", type=int, default=16, help="Number of parallel threads to use for CPU tasks")

    args = parser.parse_args()
    main(args)