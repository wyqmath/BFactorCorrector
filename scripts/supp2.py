import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import EsmTokenizer, EsmModel
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 忽略警告 ---
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置参数
# =============================================================================
TEST_DATA_PATH = "processed_data/test.jsonl"
OUTPUT_DIR = "figures/supplementary"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 模型路径 (请根据实际情况修改文件名)
MODEL_PATHS = {
    "Analytical Formula": None, 
    "XGBoost": "bfc_model/bfc_model_xgboost_bfactor_only.json",
    "ESM-2 Frozen": "bfc_model/bfc_model_esm_frozen_linear.pth",
    "BFC (Ours)": "bfc_model/bfc_model_esm_finetuned.pth"
}

# 硬件与推理参数
MAX_LEN = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 64

# =============================================================================
# 2. 模型定义
# =============================================================================
class BFCModel(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", num_extra_features=1, dropout_rate=0.1):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        esm_hidden_size = self.esm.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(esm_hidden_size + num_extra_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, other_features):
        with torch.no_grad():
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        esm_embeddings = outputs.last_hidden_state
        combined_features = torch.cat([esm_embeddings, other_features], dim=-1)
        return self.regression_head(combined_features).squeeze(-1)

# =============================================================================
# 3. 数据集与辅助函数
# =============================================================================
class EvalDataset(Dataset):
    def __init__(self, jsonl_file):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def traditional_b_to_rmsf(b_factors):
    """B-factor (Å^2) -> RMSF (Å)"""
    safe_b_factors = np.maximum(b_factors, 0)
    return np.sqrt(3 * safe_b_factors / (8 * np.pi**2))

def calculate_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2: return {'PCC': 0, 'SCC': 0, 'RMSE': 0}
    pcc, _ = pearsonr(y_true, y_pred)
    scc, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'PCC': pcc, 'SCC': scc, 'RMSE': rmse}

# =============================================================================
# 4. 统一推理引擎 (同时获取 Norm 和 Physical 结果)
# =============================================================================
def run_inference(dataset, model_name, model_path, device):
    """
    返回四个数组: pred_norm, true_norm, pred_abs, true_abs
    """
    print(f"Running inference for: {model_name}...")
    
    preds_norm_list, trues_norm_list = [], []
    preds_abs_list, trues_abs_list = [], []

    # --- A. 深度学习模型 (PyTorch) ---
    if model_name in ["ESM-2 Frozen", "BFC (Ours)"]:
        tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
        model = BFCModel(MODEL_NAME, num_extra_features=1).to(device)
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {k.replace('module.', '').replace('esm_model.', 'esm.'): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

        def collate_fn(batch):
            sequences = [item['Sequence'][:MAX_LEN] for item in batch]
            norm_b = [torch.tensor(item['Norm_B_factor'][:MAX_LEN], dtype=torch.float).unsqueeze(1) for item in batch]
            tokenized = tokenizer(sequences, padding='longest', truncation=True, max_length=MAX_LEN, return_tensors='pt', add_special_tokens=False)
            padded_feat = pad_sequence(norm_b, batch_first=True, padding_value=0.0)
            if padded_feat.shape[1] < tokenized['input_ids'].shape[1]:
                padded_feat = torch.nn.functional.pad(padded_feat, (0, 0, 0, tokenized['input_ids'].shape[1] - padded_feat.shape[1]))
            return tokenized, padded_feat, batch

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

        with torch.no_grad():
            for tokenized, features, raw_list in tqdm(loader, desc=f"Infer {model_name}"):
                input_ids = tokenized['input_ids'].to(device)
                mask = tokenized['attention_mask'].to(device)
                features = features.to(device)
                
                # Model predicts Normalized values
                batch_preds_norm = model(input_ids, mask, features).cpu().numpy()
                
                for i, raw_item in enumerate(raw_list):
                    eff_len = int(mask[i].sum())
                    
                    # 1. 获取 True Data
                    t_abs = np.array(raw_item['RMSF_true'])[:eff_len]
                    t_norm = np.array(raw_item['Norm_RMSF'])[:eff_len]
                    
                    # 2. 获取 Pred Data (Norm)
                    p_norm = batch_preds_norm[i, :eff_len]
                    
                    # 3. 转换 Pred Data (Abs)
                    mean_r = np.mean(t_abs)
                    std_r = np.std(t_abs) if np.std(t_abs) > 1e-6 else 1.0
                    p_abs = p_norm * std_r + mean_r
                    
                    preds_norm_list.append(p_norm)
                    trues_norm_list.append(t_norm)
                    preds_abs_list.append(p_abs)
                    trues_abs_list.append(t_abs)

    # --- B. XGBoost ---
    elif model_name == "XGBoost":
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
        
        for item in tqdm(dataset, desc=f"Infer {model_name}"):
            seq_len = min(len(item['Sequence']), MAX_LEN)
            t_abs = np.array(item['RMSF_true'])[:seq_len]
            t_norm = np.array(item['Norm_RMSF'])[:seq_len]
            norm_b = np.array(item['Norm_B_factor'])[:seq_len].reshape(-1, 1)
            
            # Predict Norm
            p_norm = xgb_model.predict(norm_b)
            
            # Convert to Abs
            mean_r = np.mean(t_abs)
            std_r = np.std(t_abs) if np.std(t_abs) > 1e-6 else 1.0
            p_abs = p_norm * std_r + mean_r
            
            preds_norm_list.append(p_norm)
            trues_norm_list.append(t_norm)
            preds_abs_list.append(p_abs)
            trues_abs_list.append(t_abs)

    # --- C. Analytical Formula ---
    elif model_name == "Analytical Formula":
        for item in tqdm(dataset, desc=f"Infer {model_name}"):
            seq_len = min(len(item['Sequence']), MAX_LEN)
            t_abs = np.array(item['RMSF_true'])[:seq_len]
            t_norm = np.array(item['Norm_RMSF'])[:seq_len]
            b_factors = np.array(item['B_factor'])[:seq_len]
            
            # Predict Abs directly
            p_abs = traditional_b_to_rmsf(b_factors)
            
            # Convert to Norm (Standardize prediction per protein)
            # 注意：这里我们对公式生成的物理值进行 Z-score 归一化，以便在归一化空间比较形状
            mean_p = np.mean(p_abs)
            std_p = np.std(p_abs) if np.std(p_abs) > 1e-6 else 1.0
            p_norm = (p_abs - mean_p) / std_p
            
            preds_norm_list.append(p_norm)
            trues_norm_list.append(t_norm)
            preds_abs_list.append(p_abs)
            trues_abs_list.append(t_abs)

    return (np.concatenate(preds_norm_list), np.concatenate(trues_norm_list),
            np.concatenate(preds_abs_list), np.concatenate(trues_abs_list))

# =============================================================================
# 5. 绘图逻辑
# =============================================================================
def plot_grid(data_storage, space_type, output_dir):
    """
    通用绘图函数
    space_type: 'Normalized' or 'Physical'
    """
    print(f"Generating plot for {space_type} Space...")
    
    model_order = ["Analytical Formula", "XGBoost", "ESM-2 Frozen", "BFC (Ours)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    axes = axes.flatten()
    
    # 确定坐标轴范围
    if space_type == 'Normalized':
        # 归一化空间通常在 -3 到 5 之间
        lims = [-3, 5]
        xlabel = "Ground Truth Norm. RMSF (Z-score)"
        ylabel = "Predicted Norm. RMSF (Z-score)"
        unit_rmse = ""
    else:
        # 物理空间需要根据数据最大值确定
        all_vals = []
        for m in model_order:
            if m in data_storage: all_vals.append(data_storage[m]['true_abs'])
        if all_vals:
            max_val = np.percentile(np.concatenate(all_vals), 99.5)
        else:
            max_val = 5.0
        lims = [0, max_val]
        xlabel = "Ground Truth RMSF (Å)"
        ylabel = "Predicted RMSF (Å)"
        unit_rmse = " Å"

    for i, model_name in enumerate(model_order):
        ax = axes[i]
        
        if model_name not in data_storage:
            ax.text(0.5, 0.5, "Data Not Found", ha='center')
            continue
            
        data = data_storage[model_name]
        
        if space_type == 'Normalized':
            y_pred, y_true = data['pred_norm'], data['true_norm']
        else:
            y_pred, y_true = data['pred_abs'], data['true_abs']

        # 计算指标
        metrics = calculate_metrics(y_true, y_pred)
        
        # 绘图
        hb = ax.hexbin(y_true, y_pred, gridsize=80, cmap='viridis', mincnt=5, bins='log', 
                       extent=(lims[0], lims[1], lims[0], lims[1]))
        
        ax.plot(lims, lims, 'r--', linewidth=2, alpha=0.7)
        
        ax.set_title(f"{model_name}", fontsize=16, weight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        stats_text = (f"PCC = {metrics['PCC']:.3f}\n"
                      f"SCC = {metrics['SCC']:.3f}\n"
                      f"RMSE = {metrics['RMSE']:.3f}{unit_rmse}")
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9))
        
        cb = fig.colorbar(hb, ax=ax, shrink=0.9)
        cb.set_label('log10(Count)', fontsize=10)

    plt.suptitle(f"{space_type} Space Comparison", fontsize=18, y=0.99)
    plt.tight_layout()
    
    filename = f"Supp_Figure_2_{space_type}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    print(f"Saved: {filename}")

# =============================================================================
# 6. 主程序
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = EvalDataset(TEST_DATA_PATH)
    
    data_storage = {}
    
    # 1. 运行推理，收集所有数据
    for model_name, model_path in MODEL_PATHS.items():
        try:
            if model_path and not os.path.exists(model_path):
                print(f"Warning: File not found for {model_name}. Skipping.")
                continue
                
            p_norm, t_norm, p_abs, t_abs = run_inference(dataset, model_name, model_path, DEVICE)
            
            data_storage[model_name] = {
                'pred_norm': p_norm, 'true_norm': t_norm,
                'pred_abs': p_abs,   'true_abs': t_abs
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    if not data_storage:
        print("No results generated.")
        return

    # 2. 生成归一化空间的图
    plot_grid(data_storage, 'Normalized', OUTPUT_DIR)
    
    # 3. 生成物理空间的图
    plot_grid(data_storage, 'Physical', OUTPUT_DIR)
    
    print("\nAll figures generated successfully.")

if __name__ == '__main__':
    main()