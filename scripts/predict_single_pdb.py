import os
import warnings
import json  # 新增: 用于输出JSON文件
import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
import numpy as np
from Bio.PDB import PDBParser

# --- 忽略警告，保持输出整洁 ---
warnings.filterwarnings("ignore")

# =============================================================================
# 1. 配置 (Configuration)
# =============================================================================
# --- 输入/输出路径 ---
MODEL_PATH = "bfc_model/bfc_model_esm_finetuned.pth"
PDB_FILE_PATH = "1AKI.pdb"
CHAIN_ID_TO_ANALYZE = "A"
OUTPUT_DIR = "case_study_output_esm_bfactor"
# --- 修改: 输出路径从图片改为JSON ---
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, f"{os.path.basename(PDB_FILE_PATH).split('.')[0]}_chain_{CHAIN_ID_TO_ANALYZE}_prediction_results.json")

# --- 模型参数 ---
MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 1024
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- 修复: 定义氨基酸转换字典，不再依赖Bio.PDB.Polypeptide ---
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# =============================================================================
# 2. 模型定义 (与 ablation.py 保持一致)
# =============================================================================
class BFCModel(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", num_extra_features=1, dropout_rate=0.1, freeze_esm=False):
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm_model_name)
        if freeze_esm:
            for param in self.esm.parameters():
                param.requires_grad = False
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
        predictions = self.regression_head(combined_features)
        return predictions.squeeze(-1)

# =============================================================================
# 3. PDB处理与特征提取辅助函数 (已修复)
# =============================================================================
def parse_pdb_for_seq_and_bfactor(pdb_path, chain_id):
    print(f"Parsing PDB '{pdb_path}' for sequence and B-factors of chain '{chain_id}'...")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]

    if chain_id not in model:
        raise ValueError(f"Chain '{chain_id}' not found in the PDB file.")
    chain = model[chain_id]

    sequence = []
    b_factors = []
    
    for residue in chain:
        res_name = residue.get_resname()
        # --- 修复: 使用我们自己的字典进行检查和转换 ---
        if res_name in THREE_TO_ONE and "CA" in residue:
            sequence.append(THREE_TO_ONE[res_name])
            b_factors.append(residue["CA"].get_bfactor())

    if not sequence:
        raise ValueError(f"Could not extract any valid residues from chain '{chain_id}'.")

    return "".join(sequence), np.array(b_factors)

def traditional_b_to_rmsf(b_factors):
    safe_b_factors = np.maximum(b_factors, 0)
    return np.sqrt(3 * safe_b_factors / (8 * np.pi**2))

# =============================================================================
# 4. 核心预测与保存函数 (不再绘图)
# =============================================================================
def predict_and_save_results(model, tokenizer, pdb_path, chain_id, output_json_path):
    """完整流程：加载数据 -> 预处理 -> 预测 -> 保存结果到JSON"""
    model.eval()
    
    # --- 步骤 1: 从PDB提取原始特征 ---
    try:
        sequence, b_factors_raw = parse_pdb_for_seq_and_bfactor(pdb_path, chain_id)
        seq_len = len(sequence)
        print(f"Successfully extracted sequence (length: {seq_len}) and B-factors.")
    except Exception as e:
        print(f"Error processing PDB file: {e}")
        return

    # --- 步骤 2: 特征预处理 ---
    print("Preprocessing features for model input...")
    mean_b = np.mean(b_factors_raw)
    std_b = np.std(b_factors_raw)
    std_b = 1.0 if std_b < 1e-6 else std_b
    norm_b_factor = (b_factors_raw - mean_b) / std_b
    norm_b_tensor = torch.tensor(norm_b_factor, dtype=torch.float).unsqueeze(1)

    # --- 步骤 3: Tokenization 和 Padding ---
    tokenized = tokenizer(sequence, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
    
    effective_len = min(seq_len, MAX_LEN)
    padded_features = torch.zeros(1, MAX_LEN, 1)
    padded_features[0, :effective_len, :] = norm_b_tensor[:effective_len, :]
    
    input_ids = tokenized['input_ids'].to(DEVICE)
    attention_mask = tokenized['attention_mask'].to(DEVICE)
    other_features_tensor = padded_features.to(DEVICE)

    # --- 步骤 4: 模型推理 ---
    print("Running model inference...")
    with torch.no_grad():
        prediction_norm = model(input_ids, attention_mask, other_features_tensor).cpu().squeeze(0).numpy()
    prediction_norm = prediction_norm[:effective_len]

    # --- 步骤 5: 结果后处理 ---
    prediction_physical_b = prediction_norm * std_b + mean_b
    prediction_rmsf = traditional_b_to_rmsf(prediction_physical_b)
    trad_pred_rmsf = traditional_b_to_rmsf(b_factors_raw)
    
    # --- 步骤 6: 准备并保存JSON输出 ---
    print(f"Saving prediction results to {output_json_path}...")
    
    # 构建一个结构化的字典
    output_data = {
        "pdb_id": os.path.basename(pdb_path).split('.')[0],
        "chain_id": chain_id,
        "sequence_length": effective_len,
        "sequence": sequence[:effective_len],
        "results": []
    }

    # 逐个残基添加数据
    for i in range(effective_len):
        output_data["results"].append({
            "residue_index": i + 1,
            "amino_acid": sequence[i],
            "input_b_factor": float(b_factors_raw[i]),
            "predicted_rmsf_model": float(prediction_rmsf[i]),
            "predicted_rmsf_traditional": float(trad_pred_rmsf[i])
        })
        
    # 写入文件
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("Results saved successfully!")

# =============================================================================
# 5. 主执行函数
# =============================================================================
def main():
    print("--- Single PDB Prediction (ESM-2 + B-Factor Model) ---")
    print(f"Using device: {DEVICE}")

    print(f"Loading model from {MODEL_PATH}...")
    model = BFCModel(esm_model_name=MODEL_NAME, num_extra_features=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        return
        
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    print("Model and tokenizer loaded successfully.")

    if not os.path.exists(PDB_FILE_PATH):
        print(f"Error: PDB file not found at '{PDB_FILE_PATH}'")
        return

    # --- 修改: 调用新的预测与保存函数 ---
    predict_and_save_results(model, tokenizer, PDB_FILE_PATH, CHAIN_ID_TO_ANALYZE, OUTPUT_JSON_PATH)
    
    print(f"\n--- Prediction complete! Output saved to {OUTPUT_JSON_PATH} ---")

if __name__ == '__main__':
    main()