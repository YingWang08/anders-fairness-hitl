import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 全局格式优化：严格遵循PLOS ONE强制要求，内容完全保留
plt.rcParams['font.sans-serif'] = ['Arial']  # PLOS ONE指定唯一字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10  # 保留原始字号，仅确认符合期刊规范
plt.rcParams['axes.linewidth'] = 1.0  # 统一线条宽度规范


def draw_figure1(output_file='Fig1_FE-HITL_framework.tif'):
    # 保留原始尺寸+600dpi（仅确认格式合规，内容位置不变）
    fig, ax = plt.subplots(figsize=(6.93, 4), dpi=600)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # 完全保留原始颜色、位置、文本内容
    colors = {'data': '#c2e0f0', 'human': '#f9d5c4', 'bias': '#c2e5c2', 'feedback': '#e5c2c2'}
    boxes = [
        (1, 4, 3.5, 1.5, 'Data & Fairness\nMonitoring Layer', colors['data']),
        (5.5, 4, 3.5, 1.5, 'Human Ethical\nDecision Layer', colors['human']),
        (5.5, 0.5, 3.5, 1.5, 'Bias Correction &\nOptimization Layer', colors['bias']),
        (1, 0.5, 3.5, 1.5, 'Feedback & Update\nLayer', colors['feedback'])
    ]

    # 仅优化格式：线条宽度统一为1.0pt（PLOS ONE规范），内容完全保留
    for x, y, w, h, label, color in boxes:
        rect = mpatches.Rectangle((x, y), w, h, linewidth=1.0, edgecolor='black', facecolor=color)  # 仅改linewidth为1.0
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10, linespacing=1.5)  # 内容无修改

    # 完全保留原始箭头位置，仅优化线条宽度为1.0pt
    arrows = [
        ((4.5, 4.75), (5.5, 4.75)),
        ((7.25, 4), (7.25, 2)),
        ((5.5, 1.25), (4.5, 1.25)),
        ((2.75, 2), (2.75, 4))
    ]
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))  # 仅改lw为1.0

    # 保留原始逻辑：已移除内嵌标题（符合PLOS ONE要求）
    # ax.set_title('Figure 1: FE-HITL Framework Structure', fontsize=12, weight='bold')

    # 关闭自动布局，避免篡改原始内容位置
    # plt.tight_layout()

    # 保留原始导出参数（仅确认格式合规，无修改）
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches='tight',
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()


if __name__ == '__main__':
    draw_figure1()