from curl_cffi import requests
import json
import sys
from tqdm import tqdm
import concurrent.futures
import threading

# --- 配置 ---
INPUT_FILE = 'clusterInfoTable.txt'
OUTPUT_FILE = 'all_member_ids.txt'
API_BASE_URL = 'https://pdbflex.org/php/api/PDBStats.php'
MAX_RETRIES = 5
NUM_THREADS = 16  # <-- 在这里设置线程数！

# --- 头部信息依然重要，保持原样 ---
HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
    'sec-ch-ua': '"Google Chrome";v="141", "Not?A_Brand";v="8", "Chromium";v="141"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}

def get_cluster_representatives(filename):
    """从clusterInfoTable.txt文件中读取并解析出所有cluster的代表PDB和Chain ID"""
    representatives = []
    print(f"正在从 {filename} 中读取cluster代表信息...")
    try:
        with open(filename, 'r') as f:
            next(f)
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    representatives.append((parts[0], parts[1]))
    except FileNotFoundError:
        print(f"错误: 输入文件 '{filename}' 未找到。")
        sys.exit(1)
    print(f"成功读取到 {len(representatives)} 个cluster代表。")
    return representatives

def fetch_members_for_one(session, pdb_id, chain_id, file_lock):
    """
    为单个cluster代表获取所有成员ID。
    这个函数将在单个线程中执行。
    """
    params = {'pdbID': pdb_id, 'chainID': chain_id}
    
    for attempt in range(MAX_RETRIES):
        try:
            # 使用 impersonate 参数模拟浏览器TLS指纹
            response = session.get(
                API_BASE_URL, 
                params=params, 
                timeout=30, 
                impersonate="chrome110"
            )
            response.raise_for_status()  # 如果状态码不是2xx，则抛出异常
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                members = data[0].get('otherClusterMembers')
                if members and isinstance(members, list):
                    return members  # 成功，返回成员列表
            
            return [] # 响应成功但没有成员数据

        except Exception as e:
            # tqdm.write 是线程安全的，可以在进度条存在时打印信息
            tqdm.write(f"请求 {pdb_id}{chain_id} 失败 (尝试 {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}")
            if attempt >= MAX_RETRIES - 1:
                tqdm.write(f"严重: {pdb_id}{chain_id} 在所有重试后仍然失败。")
                # 使用锁来安全地写入文件，防止多线程冲突
                with file_lock:
                    with open("failed_requests.txt", "a") as f_fail:
                        f_fail.write(f"{pdb_id}\t{chain_id}\n")
                break # 跳出重试循环
    return [] # 所有重试都失败后，返回空列表


def fetch_all_members_concurrent(representatives):
    """
    使用线程池并发地遍历所有cluster代表，调用API获取所有成员。
    """
    all_member_ids = set()
    file_lock = threading.Lock() # 创建一个文件写入锁
    
    print(f"开始使用 {NUM_THREADS} 个线程从PDBFlex API获取所有cluster的成员信息...")
    
    with requests.Session() as session:
        session.headers.update(HEADERS)
        
        # 创建一个最大工作线程数为 NUM_THREADS 的线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            # 提交所有任务到线程池
            # executor.submit会立即返回一个future对象，代表未来的计算结果
            future_to_rep = {
                executor.submit(fetch_members_for_one, session, pdb, chain, file_lock): (pdb, chain)
                for pdb, chain in representatives
            }
            
            # 使用 tqdm 和 as_completed 来实时处理已完成的任务并更新进度条
            progress_bar = tqdm(
                concurrent.futures.as_completed(future_to_rep),
                total=len(representatives),
                desc="正在处理Clusters",
                unit="cluster"
            )
            
            for future in progress_bar:
                try:
                    # 获取已完成任务的结果
                    members = future.result()
                    if members:
                        all_member_ids.update(members)
                except Exception as exc:
                    rep = future_to_rep[future]
                    progress_bar.write(f"处理 {rep[0]}{rep[1]} 时产生了一个未预料的异常: {exc}")

    return all_member_ids

def save_members_to_file(member_ids, filename):
    """将所有成员ID写入到文件中"""
    print(f"\n总共找到 {len(member_ids)} 个不重复的成员ID。")
    print(f"正在将结果写入到 {filename}...")
    with open(filename, 'w') as f:
        # 先转换为列表再排序，然后写入
        for member_id in sorted(list(member_ids)):
            f.write(f"{member_id}\n")
    print("写入完成！")


if __name__ == "__main__":
    cluster_reps = get_cluster_representatives(INPUT_FILE)
    if cluster_reps:
        # 调用并发版本的获取函数
        all_members = fetch_all_members_concurrent(cluster_reps)
        if all_members:
            save_members_to_file(all_members, OUTPUT_FILE)
        else:
            print("未能获取到任何成员ID，程序结束。")
    print("\n脚本执行完毕。")