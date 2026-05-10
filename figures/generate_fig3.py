import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（与Fig2完全一致，PLOS ONE强制要求）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 统一字号
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度

# ============================================================
# Data from revised Table 2 (30 seeds, 30% bias)
# ============================================================
variants = ['Full', 'w/o HED', 'w/o MOG', 'w/o F&U']
di = [0.864, 0.639, 0.704, 0.864]
r2 = [0.711, 0.725, 0.724, 0.711]
di_std = [0.040, 0.058, 0.049, 0.040]
r2_std = [0.009, 0.018, 0.017, 0.009]

x = np.arange(len(variants))
width = 0.25

# 适配PLOS ONE双栏尺寸（与Fig2一致：6.93in宽），dpi=600
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 统一的误差棒样式（与修正版Fig2完全一致）
error_kw = {
    'capsize': 5,          # 端线长度
    'capthick': 1.2,       # 端线粗细
    'elinewidth': 1.2,     # 误差线粗细
    'ecolor': 'black'      # 误差线颜色
}

# 绘制柱状图
rects1 = ax.bar(x - width/2, di, width, yerr=di_std, label='DI',
                edgecolor='black', linewidth=1.0, color='#4C72B0',
                error_kw=error_kw)
rects2 = ax.bar(x + width/2, r2, width, yerr=r2_std, label='R²',
                edgecolor='black', linewidth=1.0, color='#55A868',
                error_kw=error_kw)

# 坐标轴设置（与Fig2完全一致）
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9)
ax.legend(fontsize=9, frameon=False)

# 阈值线
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0)

# ============================================================
# 统计显著性标注（基于30次种子、BH校正后的ANOVA+Tukey HSD结果）
# ============================================================
# Full vs w/o HED: p < 0.001
ax.text(x[1], di[1] + di_std[1] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')
# Full vs w/o MOG: p < 0.001
ax.text(x[2], di[2] + di_std[2] + 0.03, '***', ha='center', va='bottom',
        fontsize=12, fontweight='bold', color='red')
# Full vs w/o F&U: 无显著差异，不标注

# 调整坐标轴范围
ax.set_ylim(bottom=0)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)

# 网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

plt.savefig(
    'Fig3_ablation.tif',
    dpi=600,
    bbox_inches='tight',
    format='tiff',
    pil_kwargs={"compression": "tiff_lzw"}
)
plt.close()