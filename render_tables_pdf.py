import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


SMALL_HEADERS = [
    "Algorithm",
    "|U|=100\nPerf. Ratio",
    "|U|=100\nFr. Opt.",
    "|U|=100\nRuntime (s)",
    "|U|=150\nPerf. Ratio",
    "|U|=150\nFr. Opt.",
    "|U|=150\nRuntime (s)",
    "|U|=200\nPerf. Ratio",
    "|U|=200\nFr. Opt.",
    "|U|=200\nRuntime (s)",
]

SMALL_ROWS = [
    ["greedy", "0.9981 (0.0061)", "0.90", "0.0012", "0.9980 (0.0062)", "0.90", "0.0018", "0.9931 (0.0075)", "0.30", "0.0017"],
    ["tabu", "1.0000 (0.0000)", "1.00", "0.0584", "1.0000 (0.0000)", "1.00", "0.1266", "0.9990 (0.0031)", "0.90", "0.3091"],
    ["hmetis", "0.9981 (0.0061)", "0.90", "0.0504", "0.9979 (0.0062)", "0.80", "0.0697", "0.9840 (0.0145)", "0.10", "0.1537"],
    ["hmetis_refine", "0.9981 (0.0061)", "0.90", "0.1024", "1.0000 (0.0000)", "1.00", "0.1494", "0.9952 (0.0075)", "0.60", "0.2932"],
]

HIER_HEADERS = [
    "Algorithm",
    "|U|=1200\nAvg. Cov.",
    "|U|=1200\nDelta Greedy",
    "|U|=1200\nRuntime (s)",
    "|U|=2400\nAvg. Cov.",
    "|U|=2400\nDelta Greedy",
    "|U|=2400\nRuntime (s)",
]

HIER_ROWS = [
    ["greedy", "992.5", "0.0", "0.0353", "2289.5", "0.0", "0.1355"],
    ["tabu", "1011.5", "19.0", "41.9093", "2310.0", "20.5", "334.8163"],
    ["hmetis", "970.5", "-22.0", "2.0845", "2269.0", "-20.5", "5.6964"],
    ["hmetis_refine", "999.0", "6.5", "6.7715", "2289.0", "-0.5", "11.6683"],
]


def draw_table(ax, title, headers, rows, note=None):
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        if col == 0 and row > 0:
            cell.set_text_props(weight="bold")
    if note:
        ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=9, va="bottom")


pdf_path = "mcp_interim_tables.pdf"
with PdfPages(pdf_path) as pdf:
    fig1, ax1 = plt.subplots(figsize=(14, 4.8))
    draw_table(
        ax1,
        "Table 1: Paper-style dis2 weighted MCP results",
        SMALL_HEADERS,
        SMALL_ROWS,
    )
    fig1.tight_layout()
    pdf.savefig(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12.5, 4.8))
    draw_table(
        ax2,
        "Table 2: Natural hierarchy MCP results (first 2 completed runs)",
        HIER_HEADERS,
        HIER_ROWS,
        note=(
            "At |U|=1200, hmetis_refine captures 34.2% of Tabu's gain over greedy "
            "while using 16.2% of Tabu runtime. At |U|=2400, results are interim "
            "only and based on the first 2 completed runs."
        ),
    )
    fig2.tight_layout()
    pdf.savefig(fig2)
    plt.close(fig2)

print(pdf_path)
