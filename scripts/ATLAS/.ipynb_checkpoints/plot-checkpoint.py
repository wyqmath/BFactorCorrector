import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.signal import find_peaks

# =============================================================================
# 1. 配置参数 (与您原始版本一致)
# =============================================================================
CSV_FILE = '1dci_C_prediction_normalized.csv'
TRUE_RMSF_FILE = '1dci_C_RMSF.tsv'
OUTPUT_FILE = '1dci_C_boxplot_final.png'
TERMINAL_EXCLUSION = 5
PEAK_PROMINENCE_STD = 1.0
PEAK_MIN_WIDTH = 3

# =============================================================================
# 2. 自动寻找最显著高峰的函数
# =============================================================================
def find_most_significant_peak(rmsf_file, exclusion_n, exclusion_c, prominence_std, min_width):
    """
    自动从真实的RMSF数据中识别出最显著的内部柔性高峰。
    """
    print("--- 正在自动识别最显著的高柔性区域 ---")
    try:
        df_rmsf = pd.read_csv(rmsf_file, sep='\t')
    except FileNotFoundError:
        print(f"错误: 找不到RMSF文件 '{rmsf_file}' 用于寻找高峰。")
        return None, None

    true_rmsf_avg = df_rmsf[['RMSF_R1', 'RMSF_R2', 'RMSF_R3']].mean(axis=1)
    core_rmsf = true_rmsf_avg[exclusion_n : -exclusion_c]
    core_indices = true_rmsf_avg.index[exclusion_n : -exclusion_c]
    
    prominence_threshold = core_rmsf.std() * prominence_std
    peaks, properties = find_peaks(core_rmsf, 
                                   prominence=prominence_threshold, 
                                   width=min_width)

    if len(peaks) == 0:
        print("警告: 未能在数据中找到显著的高峰。")
        return None, None

    most_prominent_peak_idx = np.argmax(properties['prominences'])
    start_idx_local = properties['left_ips'][most_prominent_peak_idx].astype(int)
    end_idx_local = properties['right_ips'][most_prominent_peak_idx].astype(int)
    
    start_res_id = core_indices[start_idx_local] + 1
    end_res_id = core_indices[end_idx_local] + 1
    
    print(f"识别完成！最显著的高柔性区域位于残基 {start_res_id} - {end_res_id}")
    return start_res_id, end_res_id

# =============================================================================
# 3. 主程序
# =============================================================================
def create_boxplot():
    """
    加载预测数据，定义高柔性区域和稳定区域，并生成箱线图。
    """
    start, end = find_most_significant_peak(
        rmsf_file=TRUE_RMSF_FILE,
        exclusion_n=TERMINAL_EXCLUSION,
        exclusion_c=TERMINAL_EXCLUSION,
        prominence_std=PEAK_PROMINENCE_STD,
        min_width=PEAK_MIN_WIDTH
    )
    if start is None:
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"错误: 找不到预测文件 '{CSV_FILE}'。")
        return

    max_residue_index = df['residue_index'].max()
    df_filtered = df[
        (df['residue_index'] > TERMINAL_EXCLUSION) &
        (df['residue_index'] <= max_residue_index - TERMINAL_EXCLUSION)
    ].copy()

    # === MODIFIED: 仅更新区域分类的名称，使其更客观 ===
    df_filtered['region_type'] = 'Stable Region'
    df_filtered.loc[
        (df_filtered['residue_index'] >= start) &
        (df_filtered['residue_index'] <= end),
        'region_type'
    ] = 'High Flexibility Region'
    
    flexible_predictions = df_filtered[df_filtered['region_type'] == 'High Flexibility Region']['predicted_norm_rmsf']
    stable_predictions = df_filtered[df_filtered['region_type'] == 'Stable Region']['predicted_norm_rmsf']
    
    if len(flexible_predictions) > 1 and len(stable_predictions) > 1:
        t_stat, p_value = ttest_ind(flexible_predictions, stable_predictions, equal_var=False)
        print(f"T-test结果: t-statistic = {t_stat:.2f}, p-value = {p_value:.2e}")
    else:
        p_value = float('nan')

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # === MODIFIED: 更新绘图顺序和调色板的键名，但保留您精心选择的颜色值 ===
    region_order = ['Stable Region', 'High Flexibility Region']
    palette = {"Stable Region": "#bdc3c7", "High Flexibility Region": "#e74c3c"}

    ax = sns.boxplot(data=df_filtered, x='region_type', y='predicted_norm_rmsf',
                     order=region_order, palette=palette, width=0.5)
    sns.stripplot(data=df_filtered, x='region_type', y='predicted_norm_rmsf',
                  order=region_order, color='0.25', alpha=0.6, jitter=0.2)

    if not np.isnan(p_value):
        y_max = df_filtered['predicted_norm_rmsf'].max()
        y_range = df_filtered['predicted_norm_rmsf'].max() - df_filtered['predicted_norm_rmsf'].min()
        
        bar_height = y_max + y_range * 0.15
        tick_height = y_max + y_range * 0.10
        text_height = bar_height + y_range * 0.05
        
        p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
        
        plt.plot([0, 0, 1, 1], [tick_height, bar_height, bar_height, tick_height], lw=1.5, c='black')
        plt.text(0.5, text_height, p_text, ha='center', va='bottom', fontsize=12, weight='bold')

        ax.set_ylim(bottom=df_filtered['predicted_norm_rmsf'].min() - y_range * 0.1,
                    top=text_height + y_range * 0.1)

    # === MODIFIED: 更新标题和标签，使其描述更精确 ===
    plt.suptitle('Predicted Flexibility of High Flexibility vs. Stable Regions', fontsize=14, weight='bold')
    plt.title('PDB ID: 1DCI', fontsize=12, weight='normal')
    
    plt.xlabel('Residue Category', fontsize=14)
    plt.ylabel('BFC Predicted Fluctuation (Normalized)', fontsize=14)
    
    # 更新X轴刻度标签以反映新的分类
    ax.set_xticklabels(['Stable', 'High Flexibility'], fontsize=12)

    plt.tight_layout()
    
    # === UNCHANGED: 保留您精心调整的布局参数 ===
    plt.subplots_adjust(top=0.9)

    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"图表已成功保存到: {OUTPUT_FILE}")

if __name__ == '__main__':
    create_boxplot()