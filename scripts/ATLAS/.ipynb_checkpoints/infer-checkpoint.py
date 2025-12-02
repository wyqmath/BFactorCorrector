import os
import pandas as pd
import numpy as np
import warnings
import torch
import torch.nn as nn
from transformers import EsmTokenizer, EsmModel

# --- 忽略警告 ---
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置参数 (Configuration)
# =============================================================================
# --- 你需要修改这里来适配不同的蛋白 ---
PDB_ID = '1dci_C'  # 你的蛋白ID，也是文件名前缀
CHAIN_ID = 'A'     # PDB文件中的链ID

# --- 模型和文件路径配置 ---
BFC_MODEL_PATH = "../bfc_model/bfc_model_esm_finetuned.pth"
ESM_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

# --- 推理参数 ---
MAX_LEN = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# =============================================================================
# 2. 数据解析函数
# =============================================================================
RESIDUE_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def parse_pdb_sequence(pdb_file, chain_id):
    sequence = []
    seen_res_ids = set()
    residues = {}
    with open(pdb_file, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            line_chain_id = line[21].strip()
            if line_chain_id == chain_id or line_chain_id == '':
                res_id = int(line[22:26])
                if res_id not in seen_res_ids:
                    seen_res_ids.add(res_id)
                    res_name = line[17:20].strip()
                    if res_name in RESIDUE_MAP:
                        residues[res_id] = RESIDUE_MAP[res_name]
    for res_id in sorted(residues.keys()):
        sequence.append(residues[res_id])
    return "".join(sequence)

def parse_tsv_data(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')
    return df

# =============================================================================
# 3. 模型定义
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
# 4. 主推理程序
# =============================================================================
def main():
    print(f"--- 开始为 {PDB_ID} 进行推理 (归一化空间) ---")

    # --- 1. 加载和准备数据 ---
    pdb_file = f"{PDB_ID}.pdb"
    bfactor_file = f"{PDB_ID}_Bfactor.tsv"
    rmsf_file = f"{PDB_ID}_RMSF.tsv"  # 仍需此文件来计算真值的归一化值
    
    try:
        sequence = parse_pdb_sequence(pdb_file, CHAIN_ID)
        bfactor_df = parse_tsv_data(bfactor_file)
        rmsf_df = parse_tsv_data(rmsf_file)
        print(f"成功从PDB文件解析出序列，长度: {len(sequence)}")
        print(f"成功从TSV文件加载B-factor数据，行数: {len(bfactor_df)}")
        print(f"成功从TSV文件加载RMSF真值数据，行数: {len(rmsf_df)}")
    except FileNotFoundError as e:
        print(f"错误：找不到文件 {e.filename}。请确保文件存在于当前目录。")
        return

    if not (len(sequence) == len(bfactor_df) == len(rmsf_df)):
        print(f"警告：数据长度不匹配！ Seq({len(sequence)}), B-factor({len(bfactor_df)}), RMSF({len(rmsf_df)})")
        return

    # 提取B-factor并进行Z-score归一化 (模型的输入)
    b_factors = bfactor_df.iloc[:, 0].values
    mean_b = np.mean(b_factors)
    std_b = np.std(b_factors) if np.std(b_factors) > 1e-6 else 1.0
    norm_b_factors = (b_factors - mean_b) / std_b

    # --- 2. 准备模型输入 ---
    tokenizer = EsmTokenizer.from_pretrained(ESM_MODEL_NAME)
    tokenized = tokenizer(sequence, return_tensors='pt', add_special_tokens=False)
    input_ids = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    features = torch.tensor(norm_b_factors, dtype=torch.float).unsqueeze(0).unsqueeze(-1).to(DEVICE)

    # --- 3. 加载BFC模型 ---
    print(f"正在从 {BFC_MODEL_PATH} 加载BFC模型...")
    model = BFCModel(ESM_MODEL_NAME, num_extra_features=1).to(DEVICE)
    try:
        state_dict = torch.load(BFC_MODEL_PATH, map_location=DEVICE)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"错误：加载模型失败！请检查路径和文件。错误信息: {e}")
        return
        
    # --- 4. 运行推理，得到预测的归一化RMSF ---
    print("开始推理...")
    with torch.no_grad():
        predicted_norm_rmsf = model(input_ids, attention_mask, features)
    predicted_norm_rmsf = predicted_norm_rmsf.cpu().numpy().flatten()
    print("推理完成！")

    # --- 5. 计算真实RMSF的归一化值，用于对比 ---
    print("正在计算真实RMSF的归一化值...")
    true_rmsf_replicates = rmsf_df[['RMSF_R1', 'RMSF_R2', 'RMSF_R3']].values
    true_rmsf_avg = np.mean(true_rmsf_replicates, axis=1)
    
    mean_true_rmsf = np.mean(true_rmsf_avg)
    std_true_rmsf = np.std(true_rmsf_avg) if np.std(true_rmsf_avg) > 1e-6 else 1.0
    
    # 执行Z-score归一化
    true_norm_rmsf = (true_rmsf_avg - mean_true_rmsf) / std_true_rmsf
    print("计算完成！")

    # --- 6. 保存归一化空间的结果 ---
    output_df = pd.DataFrame({
        'residue_index': np.arange(1, len(sequence) + 1),
        'amino_acid': list(sequence),
        'true_norm_rmsf': true_norm_rmsf,          # 真实RMSF的归一化值
        'predicted_norm_rmsf': predicted_norm_rmsf # 模型预测的归一化值
    })
    
    output_filename = f"{PDB_ID}_prediction_normalized.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"--- 推理结果已成功保存到: {output_filename} ---")
    print("文件包含了用于直接对比的归一化(Z-score)RMSF值。")


if __name__ == '__main__':
    main()