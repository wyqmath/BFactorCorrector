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
# 1. 配置参数 (Configuration)
# =============================================================================
TEST_DATA_PATH = "processed_data/test.jsonl"
OUTPUT_DIR = "figures/figure2"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# 模型路径配置
MODEL_PATHS = {
    "Analytical Formula": None, 
    "XGBoost": "bfc_model/bfc_model_xgboost_bfactor_only.json",
    "ESM-2 Frozen": "bfc_model/bfc_model_esm_frozen_linear.pth", # 名字已简化
    "BFC": "bfc_model/bfc_model_esm_finetuned.pth"               # 名字已简化
}

# 硬件与推理参数
MAX_LEN = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE = 64
SASA_THRESHOLD = 0.25  # SASA 分界线

# =============================================================================
# 2. 模型定义 (BFCModel)
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
    if len(y_true) < 2: return {'PCC': 0, 'SCC': 0, 'RMSE': 0, 'MAE': 0}
    pcc, _ = pearsonr(y_true, y_pred)
    scc, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'PCC': pcc, 'SCC': scc, 'RMSE': rmse, 'MAE': mae}

# =============================================================================
# 4. 详细推理引擎 (支持分层分析)
# =============================================================================
def eval_detailed(dataset, model_path, device, model_name):
    """
    运行推理，并收集:
    1. 预测值 (Å)
    2. 真实值 (Å)
    3. SS 标签 (用于图2c)
    4. SASA 值 (用于图2d)
    """
    print(f"Running detailed inference for {model_name}...")
    
    # --- 模型初始化 ---
    model = None
    xgb_model = None
    tokenizer = None
    
    if model_name == "XGBoost":
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(model_path)
    elif model_name not in ["Analytical Formula"]:
        tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
        model = BFCModel(MODEL_NAME, num_extra_features=1).to(device)
        state_dict = torch.load(model_path, map_location=device)
        # 修正 state_dict 键名
        new_state_dict = {k.replace('module.', '').replace('esm_model.', 'esm.'): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()

    # --- 数据容器 ---
    all_preds_abs, all_trues_abs = [], []
    all_ss, all_sasa = [], [] # 用于分层分析

    # --- 推理循环 ---
    if model is not None:
        # PyTorch 批处理
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
                
                preds_norm = model(input_ids, mask, features).cpu().numpy()
                
                for i, raw_item in enumerate(raw_list):
                    eff_len = int(mask[i].sum())
                    true_abs = np.array(raw_item['RMSF_true'])[:eff_len]
                    
                    # Norm -> Abs
                    mean_r = np.mean(true_abs)
                    std_r = np.std(true_abs) if np.std(true_abs) > 1e-6 else 1.0
                    pred_abs = preds_norm[i, :eff_len] * std_r + mean_r
                    
                    all_preds_abs.append(pred_abs)
                    all_trues_abs.append(true_abs)
                    
                    # 收集结构特征 (假设jsonl里有 'SS_Features' 和 'SASA')
                    if 'SS_Features' in raw_item:
                        all_ss.append(np.array(list(raw_item['SS_Features']))[:eff_len])
                    else:
                        all_ss.append(np.array(['-']*eff_len))
                    
                    if 'SASA' in raw_item:
                        all_sasa.append(np.array(raw_item['SASA'])[:eff_len])
                    else:
                        all_sasa.append(np.zeros(eff_len))

    else:
        # 传统公式 或 XGBoost
        for item in tqdm(dataset, desc=f"Infer {model_name}"):
            seq_len = min(len(item['Sequence']), MAX_LEN)
            true_abs = np.array(item['RMSF_true'])[:seq_len]
            
            if model_name == "Analytical Formula":
                b_factors = np.array(item['B_factor'])[:seq_len]
                pred_abs = traditional_b_to_rmsf(b_factors)
            elif model_name == "XGBoost":
                norm_b = np.array(item['Norm_B_factor'])[:seq_len].reshape(-1, 1)
                mean_r = np.mean(true_abs)
                std_r = np.std(true_abs) if np.std(true_abs) > 1e-6 else 1.0
                pred_norm = xgb_model.predict(norm_b)
                pred_abs = pred_norm * std_r + mean_r

            all_preds_abs.append(pred_abs)
            all_trues_abs.append(true_abs)
            
            # 收集结构特征
            if 'SS_Features' in item:
                all_ss.append(np.array(list(item['SS_Features']))[:seq_len])
            else:
                all_ss.append(np.array(['-']*seq_len))
            if 'SASA' in item:
                all_sasa.append(np.array(item['SASA'])[:seq_len])
            else:
                all_sasa.append(np.zeros(seq_len))
                
    return (np.concatenate(all_preds_abs), np.concatenate(all_trues_abs), 
            np.concatenate(all_ss), np.concatenate(all_sasa))

# =============================================================================
# 5. 绘图逻辑 (2x2 Grid)
# =============================================================================
def plot_figure2(benchmark_df, stratified_data, output_dir):
    print("Generating Figure 2 (2x2 Panel)...")
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ---------------------------------------------------------
    # Top Row: Benchmark (a, b) - 展示4个模型
    # ---------------------------------------------------------
    model_order = ["Analytical Formula", "XGBoost", "ESM-2 Frozen", "BFC"]
    benchmark_df['Model'] = pd.Categorical(benchmark_df['Model'], categories=model_order, ordered=True)
    benchmark_df = benchmark_df.sort_values('Model')
    
    # (a) Correlation
    corr_df = benchmark_df.melt(id_vars='Model', value_vars=['PCC', 'SCC'], var_name='Metric', value_name='Value')
    sns.barplot(data=corr_df, x='Model', y='Value', hue='Metric', ax=axes[0, 0], palette="viridis", edgecolor="black")
    axes[0, 0].set_title('a) Correlation Metrics (Physical Scale)', fontsize=14, weight='bold')
    axes[0, 0].set_ylim(0, 0.9) # 留出空间给标签
    axes[0, 0].set_xlabel('')
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].legend(loc='upper left')

    # (b) Error
    err_df = benchmark_df.melt(id_vars='Model', value_vars=['RMSE', 'MAE'], var_name='Metric', value_name='Value')
    sns.barplot(data=err_df, x='Model', y='Value', hue='Metric', ax=axes[0, 1], palette="magma_r", edgecolor="black")
    axes[0, 1].set_title('b) Error Metrics (Physical Scale)', fontsize=14, weight='bold')
    axes[0, 1].set_ylabel('Error (Å)')
    axes[0, 1].set_xlabel('')
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].legend(loc='upper right')

    # ---------------------------------------------------------
    # Bottom Row: Stratified (c, d) - 仅对比 Traditional vs BFC
    # ---------------------------------------------------------
    compare_models = ["Analytical Formula", "BFC"]
    ss_results, sasa_results = [], []
    
    for m in compare_models:
        if m not in stratified_data: continue
        preds, trues, ss, sasa = stratified_data[m]
        
        # SS Analysis (Helix, Sheet, Loop)
        for ss_type, label in [('H', 'Helix'), ('E', 'Sheet'), ('L', 'Loop')]:
            mask = (ss == ss_type) & np.isfinite(trues) & np.isfinite(preds)
            if mask.sum() > 100:
                pcc, _ = pearsonr(trues[mask], preds[mask])
                ss_results.append({'Model': m, 'Region': label, 'PCC': pcc})
        
        # SASA Analysis (Buried, Exposed)
        mask_buried = (sasa < SASA_THRESHOLD) & np.isfinite(trues) & np.isfinite(preds)
        mask_exposed = (sasa >= SASA_THRESHOLD) & np.isfinite(trues) & np.isfinite(preds)
        
        if mask_buried.sum() > 100:
            sasa_results.append({'Model': m, 'Region': 'Buried', 'PCC': pearsonr(trues[mask_buried], preds[mask_buried])[0]})
        if mask_exposed.sum() > 100:
            sasa_results.append({'Model': m, 'Region': 'Exposed', 'PCC': pearsonr(trues[mask_exposed], preds[mask_exposed])[0]})

    df_ss = pd.DataFrame(ss_results)
    df_sasa = pd.DataFrame(sasa_results)
    
    strat_palette = {"Analytical Formula": "#bdc3c7", "BFC": "#5e81d6"}

    # (c) Secondary Structure
    sns.barplot(data=df_ss, x='Region', y='PCC', hue='Model', ax=axes[1, 0], palette=strat_palette, edgecolor="black")
    axes[1, 0].set_title('c) Performance by Secondary Structure', fontsize=14, weight='bold')
    axes[1, 0].set_ylim(0, 0.9)
    axes[1, 0].set_xlabel('Structural Motif')
    axes[1, 0].legend(title=None, loc='upper left')

    # (d) Solvent Accessibility
    sns.barplot(data=df_sasa, x='Region', y='PCC', hue='Model', ax=axes[1, 1], palette=strat_palette, edgecolor="black")
    axes[1, 1].set_title('d) Performance by Solvent Accessibility', fontsize=14, weight='bold')
    axes[1, 1].set_ylim(0, 0.9)
    axes[1, 1].set_xlabel('Residue Context')
    axes[1, 1].legend(title=None, loc='upper left')

    # --- 通用标注 (Bar Labels) ---
    for ax in axes.flatten():
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height), 
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "figure2_complete.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

# =============================================================================
# 6. 主程序
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"--- Figure 2 Generation Script ---")
    
    dataset = EvalDataset(TEST_DATA_PATH)
    benchmark_metrics = []
    stratified_data = {} 

    for model_name, model_path in MODEL_PATHS.items():
        try:
            if model_path and not os.path.exists(model_path):
                print(f"Skipping {model_name}: File not found.")
                continue
                
            # 运行详细推理
            preds, trues, ss, sasa = eval_detailed(dataset, model_path, DEVICE, model_name)
            
            # 1. 计算 Benchmark 指标 (Top Row)
            metrics = calculate_metrics(trues, preds)
            metrics['Model'] = model_name
            benchmark_metrics.append(metrics)
            print(f"  -> {model_name}: PCC={metrics['PCC']:.3f}, RMSE={metrics['RMSE']:.3f}")
            
            # 2. 保存数据 (Bottom Row)
            stratified_data[model_name] = (preds, trues, ss, sasa)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")

    if benchmark_metrics:
        plot_figure2(pd.DataFrame(benchmark_metrics), stratified_data, OUTPUT_DIR)
    else:
        print("No results generated.")

if __name__ == '__main__':
    main()