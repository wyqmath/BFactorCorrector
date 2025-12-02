import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. 配置参数 (INPUT SETTINGS)
# =============================================================================
PREDICTION_FILE = '1cuo_A_prediction_normalized.csv'
GROUND_TRUTH_FILE = '1cuo_A_RMSF.tsv' # 这里存的是实验B-factor数据
OUTPUT_PLOT_NAME = '1cuo_A_comparison.png'

# 关注的区域 (Region of Interest)
ROI_START = 37
ROI_END = 44

# 标签设置
PLOT_TITLE = 'BFC Prediction vs. Experimental B-factor Profile (1CUO)'
X_LABEL = 'Residue Index'
Y_LABEL = 'Normalized Dynamics (Z-score)'
LABEL_EXP = 'Experimental B-factor'  # 代表传统方法/晶体数据
LABEL_PRED = 'BFC Prediction'        # 代表您的模型

# =============================================================================
# 2. 数据处理
# =============================================================================
def z_score_normalization(series):
    return (series - series.mean()) / series.std()

def plot_clean():
    # --- 加载数据 ---
    try:
        df_pred = pd.read_csv(PREDICTION_FILE)
        df_true = pd.read_csv(GROUND_TRUTH_FILE, sep='\t')
    except FileNotFoundError as e:
        print(e)
        return

    # 处理实验数据 (取均值或最后一列)
    cols = [c for c in df_true.columns if 'RMSF' in c or 'B_factor' in c]
    if len(cols) > 0:
        true_values = df_true[cols].mean(axis=1)
    else:
        true_values = df_true.iloc[:, -1]

    # 对齐数据长度
    residues = df_pred['residue_index']
    pred_values = df_pred['predicted_norm_rmsf']
    min_len = min(len(residues), len(true_values))
    
    # 构建绘图DataFrame
    df = pd.DataFrame({
        'Residue': residues[:min_len],
        'Exp_Z': z_score_normalization(true_values[:min_len]),
        'Pred_Z': z_score_normalization(pred_values[:min_len])
    })
    
    # 切除末端 (防止末端极其剧烈的波动干扰Y轴范围，通常切除2-3个即可)
    df = df.iloc[2:-2]

    # --- 绘图 ---
    # 设置简约风格
    sns.set_theme(style="ticks", font_scale=1.1) # 使用 ticks 风格更像传统科研绘图
    plt.figure(figsize=(9, 4.5)) # 稍微压扁一点，使线条波动更明显

    # 1. 绘制高亮背景 (61-66)
    # 使用淡淡的红色背景，不加边框
    plt.axvspan(ROI_START, ROI_END, color='#e74c3c', alpha=0.15, lw=0)

    # 2. 绘制线条
    # 实验值 (黑线): 代表"旧方法/不可靠数据"，用深灰色，稍微细一点
    sns.lineplot(data=df, x='Residue', y='Exp_Z', color='#2c3e50', linewidth=1.5, 
                 label=LABEL_EXP, alpha=0.9)
    
    # 预测值 (红线): 代表"新发现/真实物理"，用鲜艳红色，粗一点
    sns.lineplot(data=df, x='Residue', y='Pred_Z', color='#e74c3c', linewidth=2.5, 
                 label=LABEL_PRED)

    # 3. 设置坐标轴
    plt.title(PLOT_TITLE, fontsize=13, weight='bold', pad=12)
    plt.xlabel(X_LABEL, fontsize=11)
    plt.ylabel(Y_LABEL, fontsize=11)
    plt.xlim(df['Residue'].min(), df['Residue'].max())
    
    # 添加零线
    plt.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    # 4. 优化图例 (关键修改: 小一点，放角落)
    # frameon=False 去掉图例边框，显得更干净
    plt.legend(loc='upper left', fontsize=9, frameon=False, labelspacing=0.3)

    # 5. 去除多余边框 (Despine)
    sns.despine()

    # 6. 在高亮区顶部加个极小的文字标识 (可选，若不需要可注释掉)
    y_lim_max = max(df['Exp_Z'].max(), df['Pred_Z'].max())
    plt.text((ROI_START + ROI_END)/2, y_lim_max, 'Loop 37-44', 
             ha='center', va='bottom', color='#c0392b', fontsize=9, weight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_NAME, dpi=300)
    print(f"图表已保存: {OUTPUT_PLOT_NAME}")

if __name__ == '__main__':
    plot_clean()