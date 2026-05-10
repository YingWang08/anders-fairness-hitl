import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（PLOS ONE强制要求：Arial/Times New Roman）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

# ============================================================
# Data from revised Table 1 (30 seeds, 30% bias)
# ============================================================
models = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
di = [0.638, 0.654, 0.670, 0.864]
eod = [0.150, 0.136, 0.107, 0.025]
r2 = [0.736, 0.732, 0.726, 0.711]
di_std = [0.030, 0.042, 0.021, 0.040]
eod_std = [0.043, 0.045, 0.028, 0.019]
r2_std = [0.006, 0.003, 0.003, 0.009]

x = np.arange(len(models))
width = 0.25

# 适配PLOS ONE双栏尺寸，dpi=600
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 统一的误差棒样式
error_kw = {
    'capsize': 5,          # 端线长度（统一）
    'capthick': 1.2,       # 端线粗细（统一）
    'elinewidth': 1.2,     # 误差线粗细（统一）
    'ecolor': 'black'      # 误差线颜色（统一黑色）
}

# 绘制柱状图
rects1 = ax.bar(x - width, di, width, yerr=di_std, label='DI',
                edgecolor='black', linewidth=1.0, color='#4C72B0',
                error_kw=error_kw)
rects2 = ax.bar(x, eod, width, yerr=eod_std, label='EOD',
                edgecolor='black', linewidth=1.0, color='#DD8452',
                error_kw=error_kw)
rects3 = ax.bar(x + width, r2, width, yerr=r2_std, label='R²',
                edgecolor='black', linewidth=1.0, color='#55A868',
                error_kw=error_kw)

# 坐标轴设置
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=9, frameon=False)

# 阈值线
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0)

# 统计显著性标注 (FE-HITL vs DL)
ax.text(x[3], di[3] + di_std[3] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')
ax.text(x[3], eod[3] + eod_std[3] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')

# 调整坐标轴范围
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

plt.savefig(
    'Fig2_agricultural_comparison.tif',
    dpi=600,
    bbox_inches='tight',
    format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()