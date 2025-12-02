import os
import sys
import time
import requests # 使用标准的requests库，对于RCSB下载API足够且通用
import concurrent.futures
from tqdm import tqdm

# --- 配置 ---
INPUT_FILE = 'all_member_ids.txt'      # 包含所有成员ID的文件 (例如: 1A2BC, 2B3DA)
PDB_DOWNLOAD_DIR = 'pdb_files'         # PDB文件将保存到这个目录
FAILED_LOG_FILE = 'failed_pdb_downloads.txt' # 失败的下载请求记录
API_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb" # RCSB PDB文件下载API

# --- 下载参数 ---
MAX_RETRIES = 3           # 每个请求的最大重试次数
NUM_THREADS = 32          # <-- 关键参数：并发线程数，可以根据你的网络和磁盘情况调整 (16-64)
TIMEOUT = 45              # 每个下载请求的超时时间（秒）

def read_unique_pdb_ids(filename):
    """
    从输入文件中读取ID，并提取唯一的4字符PDB ID。
    例如，'1ABCA' 会被解析为 '1ABC'。
    """
    unique_pdb_ids = set()
    print(f"正在从 {filename} 中读取并解析PDB ID...")
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) >= 4:
                    # 只取前4个字符作为PDB ID
                    pdb_id = line[:4].upper() # PDB ID通常为大写
                    unique_pdb_ids.add(pdb_id)
    except FileNotFoundError:
        print(f"错误: 输入文件 '{filename}' 未找到。")
        sys.exit(1)
        
    pdb_id_list = sorted(list(unique_pdb_ids))
    print(f"成功解析到 {len(pdb_id_list)} 个唯一的PDB ID。")
    return pdb_id_list

def download_one_pdb(session, pdb_id):
    """
    为单个PDB ID下载.pdb文件。
    这个函数将在单个线程中执行。
    """
    # 构造文件路径和URL
    output_path = os.path.join(PDB_DOWNLOAD_DIR, f"{pdb_id}.pdb")
    url = API_URL_TEMPLATE.format(pdb_id=pdb_id)
    
    # --- 断点续传：如果文件已存在，则跳过 ---
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return "skipped"

    # --- 重试逻辑 ---
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=TIMEOUT)
            # 检查HTTP状态码，404 (Not Found) 等错误会触发异常
            response.raise_for_status() 
            
            # 成功下载，写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            return "success"
            
        except requests.exceptions.RequestException as e:
            # 捕获所有requests相关的错误 (连接, 超时, HTTP错误等)
            if attempt < MAX_RETRIES - 1:
                # 指数退避策略，等待后重试
                time.sleep(1.5 ** attempt) 
            else:
                # 最后一次尝试仍然失败
                tqdm.write(f"下载失败: {pdb_id} 在 {MAX_RETRIES} 次尝试后放弃。错误: {e}")
                # 将失败的ID写入日志文件
                with open(FAILED_LOG_FILE, "a") as f:
                    f.write(f"{pdb_id}\n")
                return "failed"
    return "failed" # 确保函数总有返回值

def download_all_concurrent(pdb_id_list):
    """
    使用线程池并发地下载所有PDB文件。
    """
    print(f"\n开始使用 {NUM_THREADS} 个线程从RCSB PDB下载文件...")
    print(f"文件将保存到: '{PDB_DOWNLOAD_DIR}' 目录")

    # 创建输出目录
    os.makedirs(PDB_DOWNLOAD_DIR, exist_ok=True)
    
    # 统计成功、失败和跳过的数量
    success_count = 0
    skipped_count = 0
    failed_count = 0

    # 使用Session对象进行连接复用，提高效率
    with requests.Session() as session:
        # 使用ThreadPoolExecutor进行并发下载
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # 提交所有任务
            future_to_pdb = {executor.submit(download_one_pdb, session, pdb_id): pdb_id for pdb_id in pdb_id_list}
            
            # 使用tqdm创建进度条，并处理返回结果
            pbar = tqdm(concurrent.futures.as_completed(future_to_pdb), total=len(pdb_id_list), desc="下载PDB文件", unit="file")
            for future in pbar:
                result = future.result()
                if result == "success":
                    success_count += 1
                elif result == "skipped":
                    skipped_count += 1
                else: # failed
                    failed_count += 1
                # 动态更新进度条的描述信息
                pbar.set_postfix_str(f"成功: {success_count}, 跳过: {skipped_count}, 失败: {failed_count}")

    # 返回统计结果
    return success_count, skipped_count, failed_count


if __name__ == "__main__":
    # 1. 读取并解析唯一的PDB ID
    pdb_ids_to_download = read_unique_pdb_ids(INPUT_FILE)
    
    if pdb_ids_to_download:
        start_time = time.time()
        
        # 2. 并发下载所有文件
        s, k, f = download_all_concurrent(pdb_ids_to_download)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 3. 打印最终总结
        print("\n-------------------- 下载总结 --------------------")
        print(f"总耗时: {duration:.2f} 秒")
        print(f"总计需要处理 {len(pdb_ids_to_download)} 个PDB ID。")
        print(f"  - 成功下载: {s} 个新文件")
        print(f"  - 跳过已有: {k} 个文件")
        print(f"  - 最终失败: {f} 个文件")
        
        total_files = len(os.listdir(PDB_DOWNLOAD_DIR))
        print(f"\n当前 '{PDB_DOWNLOAD_DIR}' 目录中共有 {total_files} 个PDB文件。")

        if os.path.exists(FAILED_LOG_FILE):
            print(f"失败的下载ID已记录在: {FAILED_LOG_FILE}")
            
    print("\n脚本执行完毕。")