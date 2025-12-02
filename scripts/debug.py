import json
import os

# 配置你的数据目录
DATA_DIR = 'processed_data'
FILES = ['train.jsonl', 'validation.jsonl', 'test.jsonl']

def count_stats():
    total_chains = 0
    unique_pdb_ids = set()
    
    print(f"正在统计 {DATA_DIR} 目录下的数据...")
    
    for filename in FILES:
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"警告: 文件 {filename} 不存在，跳过。")
            continue
            
        line_count = 0
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # 统计链的数量
                    total_chains += 1
                    line_count += 1
                    # 统计唯一PDB ID
                    if 'PDB_ID' in entry:
                        unique_pdb_ids.add(entry['PDB_ID'])
                except json.JSONDecodeError:
                    continue
        print(f"  - {filename}: {line_count} 条链")

    print("\n" + "="*30)
    print("【论文数据统计结果】")
    print(f"Unique PDB Entries (PDB ID): {len(unique_pdb_ids)}")
    print(f"Total Protein Chains:        {total_chains}")
    print("="*30 + "\n")
    print("请将这两个数字填入 LaTeX 的 [Insert Number] 处。")

if __name__ == "__main__":
    count_stats()