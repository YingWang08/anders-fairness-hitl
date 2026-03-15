import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（与Fig2完全一致，PLOS ONE强制要求）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 统一字号8-12pt（投稿标准）
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度（≥0.5pt）

# Data from Table 2 (ablation)
variants = ['Full', 'w/o HED', 'w/o MOG', 'w/o F&U']
di = [0.88, 0.68, 0.83, 0.79]
r2 = [0.86, 0.83, 0.85, 0.84]
di_std = [0.02, 0.03, 0.02, 0.03]
r2_std = [0.02, 0.02, 0.02, 0.02]

x = np.arange(len(variants))
width = 0.25  # 改用0.25宽度（与Fig2一致），避免柱状图过宽导致拥挤

# 适配PLOS ONE双栏尺寸（与Fig2一致：6.93in宽，4.5in高），dpi=600
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 绘制柱状图（与Fig2样式完全一致）
rects1 = ax.bar(x - width, di, width, yerr=di_std, label='DI', capsize=4,
                edgecolor='black', linewidth=1.0)  # 增加边框提升辨识度
rects2 = ax.bar(x + width, r2, width, yerr=r2_std, label='R²', capsize=4,
                edgecolor='black', linewidth=1.0)

# 坐标轴设置（与Fig2完全一致）
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=9)  # 横坐标字号略小但清晰
ax.legend(fontsize=9, frameon=False)  # 移除图例边框（与Fig2一致）

# 阈值线（0.8虚线），样式与Fig2的DI阈值线完全一致（无label避免图例冗余）
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0)

# 关键：移除图片内嵌标题（PLOS ONE禁止）
# ax.set_title('Ablation Study on Agricultural Dataset', fontsize=12)

# 调整坐标轴范围（与Fig2一致），避免内容拥挤
ax.set_ylim(bottom=0)  # Y轴从0开始，符合学术图表规范
ax.spines['top'].set_visible(True)  # 与Fig2一致，显示顶部边框
ax.spines['right'].set_visible(True)  # 与Fig2一致，显示右侧边框

plt.tight_layout()  # 自动调整布局（Fig2同款），彻底避免图例重叠

# 导出设置（与Fig2完全一致，PLOS ONE核心要求）
plt.savefig(
    'Fig3_ablation.tif',
    dpi=600,  # 线图必须≥600dpi
    bbox_inches='tight',  # 裁剪空白区域
    format='tiff',  # 优先TIFF格式，避免JPG
    pil_kwargs={"compression": "tiff_lzw"}  # 无损压缩，控制文件大小（<10MB）
)
plt.close()  # 关闭画布，释放内存
# plt.show()  # 投稿时可注释，仅需生成文件