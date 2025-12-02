from curl_cffi import requests
import pandas as pd
import sys
from tqdm import tqdm
import concurrent.futures
import threading
import time
import os

# --- 配置 ---
INPUT_FILE = 'all_member_ids.txt'      # 包含所有成员ID的文件
OUTPUT_CSV_FILE = 'rmsd_profiles.csv'   # 成功的输出文件
FAILED_LOG_FILE = 'failed_rmsd_requests.txt' # 失败的请求记录
API_BASE_URL = 'https://pdbflex.org/php/api/rmsdProfile.php'
MAX_RETRIES = 5           # 每个请求的最大重试次数
NUM_THREADS = 32          # <-- 关键参数：并发线程数，可以根据你的网络情况调整 (16-64)
TIMEOUT = 30              # 每个请求的超时时间（秒）

# --- 头部信息，保持不变 ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.9',
    'X-Requested-With': 'XMLHttpRequest',
    'Connection': 'keep-alive',
}

# 用于线程安全地写入文件的锁
csv_lock = threading.Lock()
fail_log_lock = threading.Lock()

def read_and_parse_ids(filename):
    """从输入文件中读取ID并解析成 (pdb_id, chain_id) 元组列表"""
    parsed_ids = []
    print(f"正在从 {filename} 中读取并解析PDB ID...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                # PDB ID通常是4个字符，后面是链ID
                if len(line) >= 5:
                    pdb_id = line[:4]
                    chain_id = line[4:] # 允许链ID超过1个字符
                    parsed_ids.append((pdb_id, chain_id))
    except FileNotFoundError:
        print(f"错误: 输入文件 '{filename}' 未找到。")
        sys.exit(1)
    print(f"成功解析到 {len(parsed_ids)} 个ID。")
    return parsed_ids

def fetch_rmsd_for_one(session, pdb_id, chain_id):
    """
    为单个 (pdb_id, chain_id) 获取RMSD profile。
    这个函数将在单个线程中执行。
    """
    params = {'pdbID': pdb_id, 'chainID': chain_id}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                API_BASE_URL, 
                params=params, 
                timeout=TIMEOUT, 
                impersonate="chrome110" # 使用curl-cffi模拟浏览器指纹，非常重要
            )
            response.raise_for_status()
            data = response.json()
            
            # 检查返回的数据是否有效
            if data and isinstance(data, dict) and "profile" in data:
                # 成功获取，返回包含所有信息的字典
                return {
                    "full_id": f"{pdb_id}{chain_id}",
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "rmsd_profile": data["profile"]
                }
            else:
                # API返回了成功状态码但数据格式不正确或为空
                tqdm.write(f"警告: {pdb_id}{chain_id} 返回了有效但为空或格式不正确的数据。")
                return None # 标记为失败

        except requests.errors.RequestsError as e:
            # 捕获网络相关的错误 (超时, 连接错误等)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt) # 指数退避策略
            else:
                tqdm.write(f"网络错误: {pdb_id}{chain_id} 在 {MAX_RETRIES} 次尝试后仍然失败: {e}")
                break
        except Exception as e:
            # 捕获其他所有异常 (如JSON解析错误)
            tqdm.write(f"未知错误: {pdb_id}{chain_id} 失败: {e}")
            break # 遇到解析错误等问题，通常重试无效
            
    # 如果所有尝试都失败了
    with fail_log_lock:
        with open(FAILED_LOG_FILE, "a") as f:
            f.write(f"{pdb_id}{chain_id}\n")
    return None

def process_and_save(future):
    """回调函数，在每个任务完成后被调用，用于处理结果并保存"""
    result = future.result()
    if result:
        # 使用锁确保线程安全地写入CSV文件
        with csv_lock:
            # 将字典转换为DataFrame，然后追加到CSV
            df = pd.DataFrame([result])
            # 如果文件不存在，则写入表头；否则不写表头，直接追加
            header = not os.path.exists(OUTPUT_CSV_FILE)
            df.to_csv(OUTPUT_CSV_FILE, mode='a', header=header, index=False)

def fetch_all_rmsd_concurrent(pdb_chain_list):
    """
    使用线程池并发地获取所有ID的RMSD profile。
    """
    print(f"开始使用 {NUM_THREADS} 个线程从PDBFlex API获取RMSD profiles...")
    
    # 确定已经处理过的ID，实现断点续传
    processed_ids = set()
    if os.path.exists(OUTPUT_CSV_FILE):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV_FILE)
            processed_ids = set(df_existing['full_id'])
            print(f"检测到已存在的输出文件，找到 {len(processed_ids)} 个已处理的记录。将跳过这些记录。")
        except pd.errors.EmptyDataError:
            print("输出文件为空，从头开始。")
        except Exception as e:
            print(f"读取现有CSV文件时出错: {e}，将从头开始。")

    # 过滤掉已经处理过的ID
    ids_to_fetch = [item for item in pdb_chain_list if f"{item[0]}{item[1]}" not in processed_ids]
    if not ids_to_fetch:
        print("所有ID似乎都已处理完毕！程序结束。")
        return
        
    print(f"需要处理的新记录有 {len(ids_to_fetch)} 个。")

    # 初始化CSV文件（如果不存在）
    if not os.path.exists(OUTPUT_CSV_FILE) or os.path.getsize(OUTPUT_CSV_FILE) == 0:
        pd.DataFrame(columns=["full_id", "pdb_id", "chain_id", "rmsd_profile"]).to_csv(OUTPUT_CSV_FILE, index=False)
        
    with requests.Session() as session:
        session.headers.update(HEADERS)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # 创建一个tqdm进度条
            with tqdm(total=len(ids_to_fetch), desc="获取RMSD", unit="id") as pbar:
                futures = []
                for pdb, chain in ids_to_fetch:
                    # 提交任务
                    future = executor.submit(fetch_rmsd_for_one, session, pdb, chain)
                    # 添加回调函数，当任务完成时自动保存并更新进度条
                    future.add_done_callback(lambda p: (process_and_save(p), pbar.update(1)))
                    futures.append(future)
                
                # 等待所有任务完成（虽然回调已经处理了大部分逻辑，但保留此结构确保主线程不会提前退出）
                concurrent.futures.wait(futures)


if __name__ == "__main__":
    # 1. 读取并解析ID
    pdb_list = read_and_parse_ids(INPUT_FILE)
    
    if pdb_list:
        # 2. 并发获取数据
        fetch_all_rmsd_concurrent(pdb_list)
        print("\n所有任务已提交并处理完成。")
        
        # 3. 最终统计
        try:
            final_df = pd.read_csv(OUTPUT_CSV_FILE)
            print(f"成功将 {len(final_df)} 条记录保存到 {OUTPUT_CSV_FILE}")
        except FileNotFoundError:
            print("未能生成任何输出。")

        if os.path.exists(FAILED_LOG_FILE):
            with open(FAILED_LOG_FILE, 'r') as f:
                num_failed = len(f.readlines())
            print(f"有 {num_failed} 个请求最终失败，详情请查看 {FAILED_LOG_FILE}")
            
    print("\n脚本执行完毕。")