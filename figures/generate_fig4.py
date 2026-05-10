import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（与Fig2/Fig3完全一致，PLOS ONE强制要求）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 统一字号
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度

# ============================================================
# Data from revised Table 3 (30 seeds, German Credit)
# ============================================================
models = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
di = [1.528, 1.351, 1.032, 1.032]
eod = [0.153, 0.092, 0.108, 0.108]
acc = [0.744, 0.723, 0.718, 0.718]
di_std = [0.329, 0.276, 0.221, 0.221]
eod_std = [0.091, 0.067, 0.077, 0.077]
acc_std = [0.021, 0.022, 0.021, 0.021]

# 注意：为了在上与其他指标视觉协调，可绘制 |DI - 1|
# （即距离公平线1.0的绝对距离），这里按原始DI数值绘制
x = np.arange(len(models))
width = 0.25

# 适配PLOS ONE双栏尺寸，dpi=600
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 统一的误差棒样式（与修正版Fig2/Fig3完全一致）
error_kw = {
    'capsize': 5,          # 端线长度
    'capthick': 1.2,       # 端线粗细
    'elinewidth': 1.2,     # 误差线粗细
    'ecolor': 'black'      # 误差线颜色
}

# 绘制柱状图
rects1 = ax.bar(x - width, di, width, yerr=di_std, label='DI',
                edgecolor='black', linewidth=1.0, color='#4C72B0',
                error_kw=error_kw)
rects2 = ax.bar(x, eod, width, yerr=eod_std, label='EOD',
                edgecolor='black', linewidth=1.0, color='#DD8452',
                error_kw=error_kw)
rects3 = ax.bar(x + width, acc, width, yerr=acc_std, label='Accuracy',
                edgecolor='black', linewidth=1.0, color='#55A868',
                error_kw=error_kw)

# 坐标轴设置
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=9, frameon=False, loc='upper left')

# 阈值线
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0)
ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

# 在全部 FH 和 DH 上标注 DL baseline 改善显著性
ax.text(x[3], di[3] + di_std[3] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')
ax.text(x[2], di[2] + di_std[2] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')

# 调整坐标轴范围
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

plt.savefig(
    'Fig4_credit_comparison.tif',
    dpi=600,
    bbox_inches='tight',
    format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()