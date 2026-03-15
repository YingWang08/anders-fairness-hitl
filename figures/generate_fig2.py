import matplotlib.pyplot as plt
import numpy as np
from sympy import true

# 全局字体设置（PLOS ONE强制要求：Arial/Times New Roman）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 统一字号8-12pt（投稿标准）
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度（≥0.5pt）

# Data from Table 1
models = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
di = [0.60, 0.62, 0.76, 0.88]
eod = [0.23, 0.21, 0.12, 0.05]
r2 = [0.78, 0.84, 0.75, 0.86]
di_std = [0.04, 0.03, 0.03, 0.02]
eod_std = [0.03, 0.02, 0.02, 0.01]
r2_std = [0.03, 0.02, 0.04, 0.02]

x = np.arange(len(models))
width = 0.25

# 适配PLOS ONE双栏尺寸（6.93in宽），dpi=600（线图最低要求）
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 绘制柱状图，优化误差棒样式（符合投稿清晰度要求）
rects1 = ax.bar(x - width, di, width, yerr=di_std, label='DI', capsize=4,
                edgecolor='black', linewidth=1.0)  # 增加边框提升辨识度
rects2 = ax.bar(x, eod, width, yerr=eod_std, label='EOD', capsize=4,
                edgecolor='black', linewidth=1.0)
rects3 = ax.bar(x + width, r2, width, yerr=r2_std, label='R²', capsize=4,
                edgecolor='black', linewidth=1.0)

# 坐标轴设置（简洁、清晰，符合学术图表规范）
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)  # 横坐标字号略小但清晰
ax.legend(fontsize=9, frameon=False)  # 移除图例边框更简洁

# 阈值线优化（更清晰，符合PLOS ONE样式）
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0, label='DI threshold')

# 关键：移除图片内嵌标题（PLOS ONE禁止，标题需写在正文图例中）
# ax.set_title('Fairness and Performance on Agricultural Dataset', fontsize=12)

# 调整坐标轴范围，避免内容拥挤
ax.set_ylim(bottom=0)  # Y轴从0开始，符合学术图表规范
ax.spines['top'].set_visible(True)  # 隐藏顶部边框
ax.spines['right'].set_visible(True)  # 隐藏右侧边框

plt.tight_layout()  # 自动调整布局，避免文字截断

# 导出设置（PLOS ONE核心要求）
plt.savefig(
    'Fig2_agricultural_comparison.tif',
    dpi=600,  # 线图必须≥600dpi
    bbox_inches='tight',  # 裁剪空白区域
    format='tiff',  # 优先TIFF格式，避免JPG
    pil_kwargs={"compression": "tiff_lzw"}  # 无损压缩，控制文件大小（<10MB）
)
plt.close()  # 关闭画布，释放内存
# plt.show()  # 投稿时可注释，仅需生成文件