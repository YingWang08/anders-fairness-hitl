import matplotlib.pyplot as plt
import numpy as np

# 全局字体设置（PLOS ONE强制要求：Arial/Times New Roman）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 统一字号8-12pt（投稿标准）
plt.rcParams['axes.linewidth'] = 1.0  # 坐标轴线条宽度（≥0.5pt）

# Data from Table 3
models = ['LR', 'DL', 'Debiased-HITL', 'FE-HITL']
di = [0.55, 0.58, 0.74, 0.81]
eod = [0.26, 0.24, 0.14, 0.08]
acc = [0.756, 0.781, 0.713, 0.783]
di_std = [0.04, 0.03, 0.03, 0.02]
eod_std = [0.03, 0.02, 0.02, 0.01]
acc_std = [0.012, 0.010, 0.013, 0.009]

x = np.arange(len(models))
width = 0.25

# 适配PLOS ONE双栏尺寸（6.93in宽），dpi=600（线图最低要求）
fig, ax = plt.subplots(figsize=(6.93, 4.5), dpi=600)

# 绘制柱状图，优化误差棒样式（与Fig2完全一致）
rects1 = ax.bar(x - width, di, width, yerr=di_std, label='DI', capsize=4,
                edgecolor='black', linewidth=1.0)  # 增加边框提升辨识度
rects2 = ax.bar(x, eod, width, yerr=eod_std, label='EOD', capsize=4,
                edgecolor='black', linewidth=1.0)
# 保留Accuracy%标注（核心修正项）
rects3 = ax.bar(x + width, acc, width, yerr=acc_std, label='Accuracy (%)', capsize=4,
                edgecolor='black', linewidth=1.0)

# 坐标轴设置（与Fig2完全一致）
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)  # 横坐标字号略小但清晰

# 【核心修正】图例固定到最左侧，精准定位避免重合
ax.legend(
    fontsize=9,
    frameon=False,  # 保留Fig2无边框样式
    loc='upper left',  # 基准位置设为左上
    bbox_to_anchor=(0.0, 1.0),  # 精准定位到最左侧（x=0.0）+ 顶部（y=1.0）
    handletextpad=0.2,  # 缩小图例标记与文字间距，避免占空间
    labelspacing=0.1,   # 缩小图例项垂直间距，更紧凑
    borderaxespad=0.0   # 图例与坐标轴间距清零，贴紧左侧
)

# 阈值线优化（与Fig2样式一致）
ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1.0, label='DI threshold')

# 关键：移除图片内嵌标题（PLOS ONE禁止）
# ax.set_title('Fairness and Performance on German Credit Dataset', fontsize=12)

# 调整坐标轴范围（与Fig2一致，仅保留Y轴从0开始）
ax.set_ylim(bottom=0)  # 取消自定义Y轴上限，使用matplotlib默认范围
ax.spines['top'].set_visible(True)  # 与Fig2一致，显示顶部边框
ax.spines['right'].set_visible(True)  # 与Fig2一致，显示右侧边框

# 关闭自动布局，防止篡改固定的图例位置
# plt.tight_layout()

# 导出设置（与Fig2完全一致，PLOS ONE核心要求）
plt.savefig(
    'Fig4_credit_comparison.tif',
    dpi=600,  # 线图必须≥600dpi
    bbox_inches='tight',  # 裁剪空白区域
    format='tiff',  # 优先TIFF格式，避免JPG
    pil_kwargs={"compression": "tiff_lzw"}  # 无损压缩，控制文件大小（<10MB）
)
plt.close()  # 关闭画布，释放内存
# plt.show()  # 预览时取消注释，投稿时注释