import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_figure5(output_file='Fig5_case_workflow.tif'):
    # 全局字体设置：严格遵循PLOS ONE强制格式要求
    plt.rcParams['font.sans-serif'] = ['Arial']  # PLOS ONE指定字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10  # 期刊标准字号（8-12pt）
    plt.rcParams['axes.linewidth'] = 1.0  # 统一线条宽度规范

    # PLOS ONE双栏尺寸+600dpi强制分辨率（仅调整格式，内容位置不变）
    fig, ax = plt.subplots(figsize=(6.93, 4), dpi=600)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # 完全保留原始内容：位置、文本、颜色均不修改
    steps = [
        (0.5, 6, 3, 1.5, "1. Original\nPrediction (28)", 'lightblue'),
        (4, 6, 3, 1.5, "2. Fairness\nAlert (DI=0.48)", 'lightblue'),
        (7.5, 6, 3, 1.5, "3. SHAP\nExplanation", 'lightblue'),
        (0.5, 2, 3, 1.5, "4. Multi-Options\nA:38, B:42, C:45", 'lightblue'),
        (4, 2, 3, 1.5, "5. Human Decision\nSelect C:45", 'lightblue'),
        (7.5, 2, 3, 1.5, "6. Model Update\nBias reduced", 'lightblue')
    ]

    # 仅优化格式：线条宽度统一为1.0pt（PLOS ONE规范），内容完全保留
    for x, y, w, h, text, color in steps:
        rect = patches.Rectangle((x, y), w, h,
                                 linewidth=1.0, edgecolor='black', facecolor=color)  # 仅改linewidth为1.0
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=9, linespacing=1.2)  # 字号保留原始9pt

    # 完全保留原始箭头位置，仅优化线条宽度为1.0pt
    arrow_points = [
        (3.5, 6.75, 4.0, 6.75),
        (7.0, 6.75, 7.5, 6.75),
        (2.0, 6.0, 2.0, 3.5),
        (9.0, 6.0, 9.0, 3.5),
        (3.5, 2.75, 4.0, 2.75),
        (7.0, 2.75, 7.5, 2.75),
    ]
    for start_x, start_y, end_x, end_y in arrow_points:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle='->', lw=1.0, color='black'))  # 仅改lw为1.0

    # 关闭自动布局，避免篡改原始内容位置
    # plt.tight_layout()

    # PLOS ONE强制导出格式：600dpi+TIFF+LZW无损压缩（仅优化格式，文件名保留）
    plt.savefig(
        output_file,
        dpi=600,
        bbox_inches='tight',
        format='tiff',
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()


if __name__ == '__main__':
    draw_figure5()