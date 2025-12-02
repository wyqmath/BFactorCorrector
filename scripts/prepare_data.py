# prepare_data_v4_with_NMA.py
import pandas as pd
import numpy as np
import os
import json
import ast
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor
import warnings

# --- 新增NMA依赖 ---
try:
    from prody import parsePDB, ANM, calcSqFlucts
except ImportError:
    print("错误: ProDy 库未找到。请运行 'pip install prody' 进行安装。")
    exit()

# --- 最终生产配置 ---
PDBFLEX_FILE = 'rmsd_profiles.csv'
PDB_DIR = 'pdb_files'
OUTPUT_DIR = 'processed_data' # 使用新的输出目录
RESOLUTION_CUTOFF = 3.0
MIN_CHAIN_LENGTH = 50
NUM_WORKERS = 32
TEST_SET_SIZE = 10000 

# --- NMA 配置 ---
NMA_MODE_CUTOFF = 20  # 使用前20个非平凡模式进行计算，这是一个常见的权衡选择

# --- 内置映射 ---
THREE_TO_ONE = { 'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V' }
NON_STANDARD_MAP = { "MSE": "MET" }
STANDARD_AMINO_ACIDS = set(THREE_TO_ONE.keys())

# --- 初始化 ---
# 注意：Bio.PDB的PDBParser在这里可以不再需要，因为ProDy的解析更强大且对NMA友好
# parser = PDBParser(QUIET=True) 
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fast_check_pdb_v2(pdb_file_path):
    # (此函数保持不变)
    resolution, is_xray = None, False
    try:
        with open(pdb_file_path, 'r', errors='ignore') as f:
            for line in f:
                if resolution is not None and is_xray: break
                if line.startswith('ATOM'): break
                line_upper = line.upper()
                if line.startswith(('REMARK   2', 'REMARK   3')) and 'RESOLUTION' in line_upper:
                    if resolution is not None: continue
                    parts = line.split()
                    for part in parts:
                        try:
                            res_val = float(part)
                            if 0.5 < res_val < 15.0:
                                resolution = res_val
                                break
                        except ValueError:
                            continue
                elif line.startswith('EXPDTA') and 'X-RAY DIFFRACTION' in line_upper:
                    is_xray = True
    except FileNotFoundError:
        return None, False
    return resolution, is_xray

def process_chain_worker(args):
    _, pdb_id_lc, chain_id, rmsf_array_str = args
    pdb_id_uc = pdb_id_lc.upper()
    pdb_file_path = os.path.join(PDB_DIR, f'{pdb_id_uc}.pdb')

    resolution, is_xray = fast_check_pdb_v2(pdb_file_path)
    if not is_xray or resolution is None or resolution > RESOLUTION_CUTOFF:
        return None
    
    try:
        # --- 使用ProDy进行解析和NMA计算 ---
        # 1. 解析PDB并选择C-alpha原子
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            protein = parsePDB(pdb_file_path)
        
        # 兼容多模型PDB，只用第一个模型
        if protein.numModels() > 1:
            protein = protein.select('model 1')
            
        ca_chain = protein.select(f'calpha and chain {chain_id}')
        if ca_chain is None or ca_chain.numAtoms() < MIN_CHAIN_LENGTH:
            return None

        # 2. 执行NMA
        anm = ANM(f'{pdb_id_uc}_{chain_id}')
        anm.buildHessian(ca_chain)
        # 加上模式数量限制，避免计算所有模式，加快速度
        # +6 是因为前6个模式是平凡的（刚体平动和转动）
        anm.calcModes(n_modes=NMA_MODE_CUTOFF + 6)
        
        # 3. 从NMA计算理论涨落
        # 使用前NMA_MODE_CUTOFF个非平凡模式
        nma_fluctuations = calcSqFlucts(anm[:NMA_MODE_CUTOFF])

        # --- 沿用BioPython/DSSP获取其他特征 ---
        # 这里的DSSP部分可以保留，但需要用Bio.PDB重新解析一次，或者想办法从ProDy对象映射
        # 为了简单起见，我们还是用Bio.PDB来处理DSSP和B-factor提取
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id_lc, pdb_file_path)
        model = structure[0]
        chain = model[chain_id]
        dssp = DSSP(model, pdb_file_path, dssp='mkdssp')
    except Exception:
        return None

    chain_data = []
    # 确保ProDy选出的C-alpha原子和DSSP/BioPython的残基能对应上
    # ProDy的ca_chain.getResindices() 和 BioPython的residue.get_id()[1] 应该是一致的
    bio_residues = [res for res in chain.get_residues()]
    
    # 一个简单的健全性检查
    if len(bio_residues) < ca_chain.numAtoms() or len(nma_fluctuations) != ca_chain.numAtoms():
        # 如果长度不匹配，说明结构有问题或解析不一致，跳过
        return None
        
    prody_resindices = ca_chain.getResindices()
    bio_residue_map = {res.get_id()[1]: res for res in bio_residues}

    for i, res_idx in enumerate(prody_resindices):
        if res_idx not in bio_residue_map:
            continue
        residue = bio_residue_map[res_idx]
        
        resname = residue.get_resname()
        if resname in NON_STANDARD_MAP:
            resname = NON_STANDARD_MAP[resname]
        
        if resname in STANDARD_AMINO_ACIDS and 'CA' in residue:
            res_key = (chain.id, residue.get_id())
            if res_key in dssp:
                b_factor = residue['CA'].get_bfactor()
                ss_full, asa = dssp[res_key][2], dssp[res_key][3]
                
                if isinstance(asa, (int, float)):
                    ss_simplified = ss_full if ss_full in ['H', 'E'] else 'L'
                    chain_data.append({ 
                        'B_factor': b_factor, 
                        'Sequence': resname, 
                        'SS': ss_simplified, 
                        'SASA': asa,
                        'NMA_Fluctuation': nma_fluctuations[i] # 添加NMA特征
                    })

    pdb_true_len = len(chain_data)
    if pdb_true_len < MIN_CHAIN_LENGTH:
        return None

    rmsf_array_full = np.array(ast.literal_eval(rmsf_array_str))
    if len(rmsf_array_full) < pdb_true_len:
        return None
    
    rmsf_array_truncated = rmsf_array_full[:pdb_true_len]
    
    for i in range(pdb_true_len):
        chain_data[i]['RMSF_true'] = rmsf_array_truncated[i]
        
    df_chain = pd.DataFrame(chain_data)
    
    if df_chain.empty or 'B_factor' not in df_chain.columns or 'RMSF_true' not in df_chain.columns or 'NMA_Fluctuation' not in df_chain.columns:
        return None
        
    # --- 标准化所有数值特征 ---
    b_mean, b_std = df_chain['B_factor'].mean(), df_chain['B_factor'].std()
    rmsf_mean, rmsf_std = df_chain['RMSF_true'].mean(), df_chain['RMSF_true'].std()
    nma_mean, nma_std = df_chain['NMA_Fluctuation'].mean(), df_chain['NMA_Fluctuation'].std()

    if b_std < 1e-6 or rmsf_std < 1e-6 or nma_std < 1e-6:
        return None
        
    df_chain['Norm_B_factor'] = (df_chain['B_factor'] - b_mean) / b_std
    df_chain['Norm_RMSF'] = (df_chain['RMSF_true'] - rmsf_mean) / rmsf_std
    df_chain['Norm_NMA_Fluctuation'] = (df_chain['NMA_Fluctuation'] - nma_mean) / nma_std
    
    return {
        'full_id': f'{pdb_id_lc}:{chain_id}', 'PDB_ID': pdb_id_lc, 'Chain_ID': chain_id, 'Length': pdb_true_len,
        'Sequence': "".join([THREE_TO_ONE.get(r, 'X') for r in df_chain['Sequence']]),
        'Norm_B_factor': df_chain['Norm_B_factor'].tolist(), 
        'SS_Features': df_chain['SS'].tolist(),
        'SASA': df_chain['SASA'].tolist(),
        'Norm_NMA_Fluctuation': df_chain['Norm_NMA_Fluctuation'].tolist(), # 新增归一化NMA特征
        'Norm_RMSF': df_chain['Norm_RMSF'].tolist(),
        # --- 保存原始数据 ---
        'B_factor': df_chain['B_factor'].tolist(),
        'RMSF_true': df_chain['RMSF_true'].tolist(),
        'NMA_Fluctuation': df_chain['NMA_Fluctuation'].tolist() # 新增原始NMA特征
    }

def main():
    # (此函数主干逻辑保持不变，除了输出目录和日志信息)
    print("--- 最终数据处理脚本 (生产版 V4 - 集成NMA特征) ---")
    print(f"配置: {NUM_WORKERS} 进程, 分辨率 < {RESOLUTION_CUTOFF} Å, 最小长度 > {MIN_CHAIN_LENGTH}")
    print(f"NMA配置: 使用前 {NMA_MODE_CUTOFF} 个非平凡模式。")
    
    df_rmsf = pd.read_csv(PDBFLEX_FILE)
    print(f"总计 {len(df_rmsf)} 条链待处理...")

    unique_pdb_ids = df_rmsf['pdb_id'].unique()
    
    np.random.seed(42)
    np.random.shuffle(unique_pdb_ids)
    
    if len(unique_pdb_ids) < TEST_SET_SIZE * 2:
        raise ValueError(f"数据集太小 ({len(unique_pdb_ids)} IDs)，无法创建大小为 {TEST_SET_SIZE} 的测试集。")

    test_ids = unique_pdb_ids[:TEST_SET_SIZE]
    remaining_ids = unique_pdb_ids[TEST_SET_SIZE:]
    train_ids, val_ids = train_test_split(remaining_ids, test_size=0.2, random_state=42)

    print(f"数据划分完成: {len(train_ids)} PDBs (训练), {len(val_ids)} PDBs (验证), {len(test_ids)} PDBs (测试)")
    
    datasets = { 
        'train': df_rmsf[df_rmsf['pdb_id'].isin(train_ids)], 
        'validation': df_rmsf[df_rmsf['pdb_id'].isin(val_ids)],
        'test': df_rmsf[df_rmsf['pdb_id'].isin(test_ids)]
    }
    
    for name, df in datasets.items():
        output_path = os.path.join(OUTPUT_DIR, f'{name}.jsonl')
        print(f"\n--- 正在处理 {name} 集 ({len(df)} 条链)...")
        tasks = df.to_records(index=False)
        processed_count = 0
        with open(output_path, 'w') as f, ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # 使用tqdm包装executor.map来显示进度条
            results = tqdm(executor.map(process_chain_worker, tasks), total=len(df), desc=f"Processing {name}")
            for result in results:
                if result:
                    json.dump(result, f)
                    f.write('\n')
                    processed_count += 1
        
        pass_rate = (processed_count / len(df)) * 100 if len(df) > 0 else 0
        print(f"成功处理并写入 {name} 集链条数: {processed_count} / {len(df)} ({pass_rate:.2f}%)")

    print(f"\n--- 数据处理完成！---")
    print(f"高质量数据集已保存至 '{OUTPUT_DIR}' 目录。")
    print("现在可以训练包含NMA特征的增强版模型。")

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        main()