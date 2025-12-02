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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 忽略警告，保持输出整洁 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# 1. 硬编码配置 (Hardcoded Configuration)
# =============================================================================
# 确保路径指向消融实验生成的模型
MODEL_PATH = "bfc_model/bfc_model_esm_finetuned.pth" 
TEST_DATA_PATH = "processed_data/test.jsonl"
OUTPUT_DIR = "figures/figure1"
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- 模型参数 ---
MAX_LEN = 1024  # 必须与 ablation.py 训练时的 max_len 一致
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 8
BATCH_SIZE_INFERENCE = 64

# --- 绘图参数 ---
NUM_CASE_STUDIES = 6
CASE_STUDY_HIGHLIGHT_ID = "1ubq_A" 

# =============================================================================
# 2. 模型定义 (必须与 ablation.py 完全一致)
# =============================================================================
class BFCModel(nn.Module):
    """
    B-Factor Corrector (BFC) Model.
    Copied strictly from ablation.py to ensure state_dict compatibility.
    """
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", num_extra_features=1, dropout_rate=0.1, freeze_esm=False):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        
        # 即使 freeze_esm=False (微调模式)，模型结构也是一样的，可以直接加载
        esm_hidden_size = self.esm.config.hidden_size
        
        self.regression_head = nn.Sequential(
            nn.Linear(esm_hidden_size + num_extra_features, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, other_features):
        with torch.no_grad(): # 推理时不需要计算梯度
             outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        
        esm_embeddings = outputs.last_hidden_state
        # other_features 只包含 Norm_B (Shape: [Batch, Len, 1])
        combined_features = torch.cat([esm_embeddings, other_features], dim=-1)
        predictions = self.regression_head(combined_features)
        return predictions.squeeze(-1)

# =============================================================================
# 3. 数据集与辅助函数
# =============================================================================
class FigureGenDataset(Dataset):
    """
    加载测试数据。
    根据消融实验结果，这里只加载 Sequence 和 Norm_B_factor。
    """
    def __init__(self, jsonl_file):
        print(f"Loading data for figure generation from {jsonl_file}...")
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item['Sequence']
        
        # 仅使用 Norm_B_factor，维度 [SeqLen, 1]
        norm_b = torch.tensor(item['Norm_B_factor'], dtype=torch.float).unsqueeze(1)
        
        return {
            'sequence': sequence,
            'other_features': norm_b, 
            'raw_data': item 
        }

def traditional_b_to_rmsf(b_factors):
    """传统公式: B-factor (Å^2) -> RMSF (Å)"""
    safe_b_factors = np.maximum(b_factors, 0)
    return np.sqrt(3 * safe_b_factors / (8 * np.pi**2))

def calculate_metrics(y_true, y_pred):
    """计算 PCC, SCC, RMSE, MAE"""
    is_valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[is_valid], y_pred[is_valid]
    
    if len(y_true) < 2:
        return {'PCC': 0, 'SCC': 0, 'RMSE': float('inf'), 'MAE': float('inf')}
        
    pcc, _ = pearsonr(y_true, y_pred)
    scc, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {'PCC': pcc, 'SCC': scc, 'RMSE': rmse, 'MAE': mae}

# =============================================================================
# 4. 推理函数
# =============================================================================
def get_all_predictions(model, data_loader, device):
    model.eval()
    all_results = []
    
    pbar = tqdm(data_loader, desc="Running Inference")
    for batch in pbar:
        tokenized, other_features_padded, raw_data_list = batch
        
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        other_features_padded = other_features_padded.to(device)

        with torch.no_grad():
            # 模型输出 Normalized RMSF
            predictions_norm = model(input_ids, attention_mask, other_features_padded).cpu().numpy()

        for i in range(len(raw_data_list)):
            raw_item = raw_data_list[i]
            original_seq_len = len(raw_item['Sequence'])
            effective_seq_len = min(original_seq_len, MAX_LEN)

            if effective_seq_len == 0: continue

            # 获取数据切片
            pred_norm = predictions_norm[i, :effective_seq_len]
            true_norm_rmsf = np.array(raw_item['Norm_RMSF'])[:effective_seq_len]
            true_norm_b = np.array(raw_item['Norm_B_factor'])[:effective_seq_len]
            true_unnorm_rmsf = np.array(raw_item['RMSF_true'])[:effective_seq_len]
            true_unnorm_b = np.array(raw_item['B_factor'])[:effective_seq_len]

            # 反归一化参数
            mean_rmsf = np.mean(true_unnorm_rmsf)
            std_rmsf = np.std(true_unnorm_rmsf)
            if std_rmsf < 1e-6: std_rmsf = 1.0
            
            # 1. 模型预测: Norm -> Abs
            pred_unnorm = pred_norm * std_rmsf + mean_rmsf
            
            # 2. 传统公式: Abs B -> Abs RMSF
            trad_pred_unnorm = traditional_b_to_rmsf(true_unnorm_b)
            
            # 3. 传统公式的 Norm 版本 (用于 Fig 1b 对比)
            norm_trad_pred = (trad_pred_unnorm - mean_rmsf) / std_rmsf

            all_results.append({
                'id': raw_item.get('PDB_ID', f'prot_{i}'),
                'norm_pred': pred_norm, 
                'norm_rmsf': true_norm_rmsf, 
                'norm_b': true_norm_b,
                'abs_pred': pred_unnorm, 
                'abs_rmsf': true_unnorm_rmsf, 
                'abs_b': true_unnorm_b,
                'abs_trad_pred': trad_pred_unnorm,
                'norm_trad_pred': norm_trad_pred,
            })
            
    return all_results

# =============================================================================
# 5. 绘图函数
# =============================================================================
def plot_figure_1b_combined(all_results, output_dir):
    """
    [修改] 图 1b: 归一化空间的 Hexbin 对比。
    现在包含 PCC, SCC, RMSE 三个指标。
    """
    print("Generating Figure 1b: Performance in Normalized Space (with SCC)...")
    
    all_norm_b = np.concatenate([r['norm_b'] for r in all_results])
    all_norm_rmsf = np.concatenate([r['norm_rmsf'] for r in all_results])
    all_norm_pred = np.concatenate([r['norm_pred'] for r in all_results])
    
    metrics_input = calculate_metrics(all_norm_rmsf, all_norm_b)
    metrics_model = calculate_metrics(all_norm_rmsf, all_norm_pred)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True, sharey=True)
    
    # 设定显示范围，排除极少数离群点以优化显示效果
    lims = [np.percentile(all_norm_rmsf, 0.5), np.percentile(all_norm_rmsf, 99.5)]
    
    # --- 左图：输入信号 (Input Signal) ---
    ax = axes[0]
    hb1 = ax.hexbin(all_norm_b, all_norm_rmsf, gridsize=100, cmap='viridis', mincnt=10, bins='log', extent=(lims[0], lims[1], lims[0], lims[1]))
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_title("Input Signal (B-Factor)", fontsize=16, weight='bold')
    ax.set_xlabel("Normalized B-Factor (Z-score)", fontsize=14)
    ax.set_ylabel("Ground Truth Normalized RMSF (Z-score)", fontsize=14)
    
    # [修改] 添加 SCC
    stats_text_1 = (f"PCC = {metrics_input['PCC']:.3f}\n"
                    f"SCC = {metrics_input['SCC']:.3f}\n"
                    f"RMSE = {metrics_input['RMSE']:.3f}")
    
    ax.text(0.05, 0.95, stats_text_1, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    fig.colorbar(hb1, ax=ax, label='log10(count)')

    # --- 右图：模型预测 (Model Prediction) ---
    ax = axes[1]
    hb2 = ax.hexbin(all_norm_pred, all_norm_rmsf, gridsize=100, cmap='viridis', mincnt=10, bins='log', extent=(lims[0], lims[1], lims[0], lims[1]))
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_title("BFC-Model Prediction", fontsize=16, weight='bold')
    ax.set_xlabel("Predicted Normalized RMSF (Z-score)", fontsize=14)
    
    # [修改] 添加 SCC
    stats_text_2 = (f"PCC = {metrics_model['PCC']:.3f}\n"
                    f"SCC = {metrics_model['SCC']:.3f}\n"
                    f"RMSE = {metrics_model['RMSE']:.3f}")

    ax.text(0.05, 0.95, stats_text_2, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.9))
    fig.colorbar(hb2, ax=ax, label='log10(count)')

    for ax in axes:
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "figure_1b_normalized_comparison.png")
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"  - Saved: {save_path}")

def plot_physical_space_scatters(all_results, output_dir):
    """图: 物理尺度 (Å) 下的 Hexbin 对比"""
    print("Generating Physical Space plots...")
    
    all_abs_rmsf = np.concatenate([r['abs_rmsf'] for r in all_results])
    all_abs_pred = np.concatenate([r['abs_pred'] for r in all_results])
    all_abs_trad_pred = np.concatenate([r['abs_trad_pred'] for r in all_results])
    
    mask = np.isfinite(all_abs_rmsf) & np.isfinite(all_abs_pred) & np.isfinite(all_abs_trad_pred)
    y_true, y_pred_model, y_pred_trad = all_abs_rmsf[mask], all_abs_pred[mask], all_abs_trad_pred[mask]

    metrics_model = calculate_metrics(y_true, y_pred_model)
    metrics_trad = calculate_metrics(y_true, y_pred_trad)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    lim_max = np.percentile(y_true, 99.5)
    lims = [0, lim_max]

    # 左图：传统公式
    ax = axes[0]
    hb = ax.hexbin(y_true, y_pred_trad, gridsize=100, cmap='viridis', mincnt=10, bins='log', extent=(0, lim_max, 0, lim_max))
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_title(f'Traditional Formula\n(PCC = {metrics_trad["PCC"]:.3f}, RMSE = {metrics_trad["RMSE"]:.3f} Å)', fontsize=15)
    ax.set_xlabel('True RMSF (Å)', fontsize=14)
    ax.set_ylabel('Predicted RMSF (Å)', fontsize=14)
    fig.colorbar(hb, ax=ax, label='log10(count)')

    # 右图：我们的模型
    ax = axes[1]
    hb = ax.hexbin(y_true, y_pred_model, gridsize=100, cmap='viridis', mincnt=10, bins='log', extent=(0, lim_max, 0, lim_max))
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_title(f'BFC-Model (Ours)\n(PCC = {metrics_model["PCC"]:.3f}, RMSE = {metrics_model["RMSE"]:.3f} Å)', fontsize=15)
    ax.set_xlabel('True RMSF (Å)', fontsize=14)
    ax.set_ylabel('Predicted RMSF (Å)', fontsize=14)
    fig.colorbar(hb, ax=ax, label='log10(count)')

    for ax in axes:
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_physical_space_comparison.png"), dpi=300); plt.close()

def plot_figure_1c(all_results, output_dir):
    """图 1c: 指标对比柱状图"""
    print("Generating Figure 1c: Metrics Bar Plot...")
    
    all_abs_rmsf = np.concatenate([r['abs_rmsf'] for r in all_results])
    all_abs_pred = np.concatenate([r['abs_pred'] for r in all_results])
    all_abs_trad_pred = np.concatenate([r['abs_trad_pred'] for r in all_results])
    
    metrics_model = calculate_metrics(all_abs_rmsf, all_abs_pred)
    metrics_trad = calculate_metrics(all_abs_rmsf, all_abs_trad_pred)

    df_data = {
        'Metric': ['PCC', 'SCC', 'RMSE [Å]', 'MAE [Å]'],
        'Traditional Formula': [metrics_trad['PCC'], metrics_trad['SCC'], metrics_trad['RMSE'], metrics_trad['MAE']],
        'BFC-Model (Ours)': [metrics_model['PCC'], metrics_model['SCC'], metrics_model['RMSE'], metrics_model['MAE']]
    }
    df = pd.DataFrame(df_data).melt(id_vars='Metric', var_name='Method', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x='Metric', y='Value', hue='Method', palette=['gray', 'firebrick'])
    
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', fontsize=10, weight='bold')

    plt.title("Performance Comparison (Physical Scale)", fontsize=16, weight='bold')
    plt.legend(title=None, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "figure_1c_metrics.png"), dpi=300); plt.close()

def plot_figure_1d_case_studies(all_results, num_cases, output_dir):
    """图 1d: 挑选改善最明显的案例展示"""
    print(f"Generating Figure 1d: Top {num_cases} case studies...")
    
    case_metrics = []
    for r in all_results:
        if len(r['abs_rmsf']) < 20: continue
        try:
            pcc_model, _ = pearsonr(r['abs_rmsf'], r['abs_pred'])
            pcc_trad, _ = pearsonr(r['abs_rmsf'], r['abs_trad_pred'])
            if not np.isnan(pcc_model) and not np.isnan(pcc_trad):
                pcc_improvement = pcc_model - pcc_trad
                case_metrics.append({'id': r['id'], 'improvement': pcc_improvement})
        except:
            continue
    
    if not case_metrics:
        print("No valid cases found.")
        return

    df_metrics = pd.DataFrame(case_metrics).sort_values(by='improvement', ascending=False)
    top_cases_ids = df_metrics.head(num_cases)['id'].tolist()
    
    results_dict = {r['id']: r for r in all_results}
    
    for i, case_id in enumerate(top_cases_ids):
        case_data = results_dict.get(case_id)
        if not case_data: continue

        residue_indices = np.arange(1, len(case_data['norm_rmsf']) + 1)
        
        plt.figure(figsize=(12, 5))
        sns.set_style("ticks")
        
        # 1. 输入信号
        plt.plot(residue_indices, case_data['norm_b'], 'o--', color='gray', markersize=4, alpha=0.5, label='Input: Norm B-Factor')
        # 2. 真实值
        plt.plot(residue_indices, case_data['norm_rmsf'], '-', color='royalblue', linewidth=2.5, label='Ground Truth RMSF')
        # 3. 传统方法
        plt.plot(residue_indices, case_data['norm_trad_pred'], ':', color='seagreen', linewidth=2, alpha=0.8, label='Traditional Formula')
        # 4. 模型预测
        plt.plot(residue_indices, case_data['norm_pred'], '--', color='crimson', linewidth=2.5, label='BFC-Model Prediction')
        
        if CASE_STUDY_HIGHLIGHT_ID.lower() in case_id.lower():
            plt.axvspan(4, 12, color='yellow', alpha=0.2, label='Flexible Loop Region')
        
        imp = df_metrics[df_metrics['id'] == case_id]['improvement'].iloc[0]
        plt.title(f"Case Study: {case_id} (PCC Improvement: +{imp:.3f})", fontsize=16, weight='bold')
        plt.xlabel("Residue Number", fontsize=12)
        plt.ylabel("Relative Flexibility (Z-score)", fontsize=12)
        plt.legend(loc='upper right', framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        safe_name = case_id.replace(":", "_")
        plt.savefig(os.path.join(output_dir, f"figure_1d_case_{i+1}_{safe_name}.png"), dpi=300); plt.close()

# =============================================================================
# 6. 主程序
# =============================================================================
def main():
    print("--- Starting Figure Generation ---")
    
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")

    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = BFCModel(MODEL_NAME, num_extra_features=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully.")
    
    # 2. 准备数据
    dataset = FigureGenDataset(TEST_DATA_PATH)
    
    def collate_fn(batch):
        sequences = [item['sequence'][:MAX_LEN] for item in batch]
        other_features_list = [item['other_features'][:MAX_LEN] for item in batch]
        raw_data_list = [item['raw_data'] for item in batch]
        
        tokenized = tokenizer(sequences, padding='longest', truncation=True, max_length=MAX_LEN, return_tensors='pt', add_special_tokens=False)
        
        # Pad B-factor
        other_features_padded = pad_sequence(other_features_list, batch_first=True, padding_value=0.0)
        
        # 对齐 Sequence 长度
        max_batch_len = tokenized['input_ids'].shape[1]
        if other_features_padded.shape[1] < max_batch_len:
            pad_len = max_batch_len - other_features_padded.shape[1]
            other_features_padded = torch.nn.functional.pad(other_features_padded, (0, 0, 0, pad_len), value=0.0)
            
        return tokenized, other_features_padded, raw_data_list

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    
    # 3. 运行推理
    all_results = get_all_predictions(model, data_loader, device)
    
    # 4. 生成图表
    if not all_results:
        print("No results generated.")
        return

    plot_figure_1b_combined(all_results, OUTPUT_DIR)
    plot_physical_space_scatters(all_results, OUTPUT_DIR)
    plot_figure_1c(all_results, OUTPUT_DIR)
    plot_figure_1d_case_studies(all_results, NUM_CASE_STUDIES, OUTPUT_DIR)
    
    print("\n--- Figure generation complete! ---")

if __name__ == '__main__':
    main()