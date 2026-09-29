# -*- coding: utf-8 -*-
"""
Bacterial growth curve fitting with Logistic model and CNS-style plotting.

修订说明 / Revision notes (2026-09-24)
-------------------------------------
- 沿用原始数据、Logistic 拟合、重复测量点和均值 ± 样本标准差。
- 使用线性纵轴呈现 Logistic S 形，保留蓝/橙/绿/红配色和中文坐标轴。
- 保留交点时间标注；移除仅适用于对数纵轴的 10× 比例尺。
- 不绘制 2×10⁸ CFU/mL 水平阈值线及其文字说明。
- 固定 16 × 9 英寸画布，以 600 dpi 输出 9600 × 5400 像素 PNG。
- 保存时不自动裁边，确保文件的宽高比严格为 16:9。
- 图片与拟合结果保存到脚本所在目录；命令行 --no-show 可关闭弹窗。

@author: luoh
"""
import logging
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import curve_fit, fsolve
import pandas as pd


# =============================================================================
# 常量 / Constants
# =============================================================================
COLOR_MAP = plt.cm.tab10(np.linspace(0, 1, 10))

# ---- 高清导出：尺寸和 DPI 可在此统一调整 ----
FIGURE_SIZE = (16, 9)                     # 英寸，严格 16:9
EXPORT_DPI = 600                         # PNG: 9600 × 5400 像素

def darken_color(c, factor=0.55):
    """将颜色变深 (与OD深色点版本相同)"""
    return tuple(min(1.0, x * factor) for x in c[:3]) + (c[3] if len(c) > 3 else 1.0,)

# 样品原始 key → 图注中文名
NAME_DISPLAY_CN: Dict[str, str] = {
    '1e1': '10¹',
    '1e2': '10²',
    '1e3': '10³',
    '1e4': '10⁴',
}

# ---- 轴布局 ----
X_AXIS_PADDING = 1.0                       # max(time_points) 之后留的小时数
Y_AXIS_LOWER_LIMIT = 0.0                   # 线性纵轴从零开始，呈现完整 S 形
Y_AXIS_UPPER_FACTOR = 1.12                 # 为最高数据点及误差棒留出空间

# ---- 交点搜索 ----
INTERSECTION_T_MIN = 0.0                   # h
INTERSECTION_T_MAX_DEFAULT = 100.0         # h(默认上限,允许外推)
INTERSECTION_TOLERANCE = 1e-6              # CFU/mL 残差
INTERSECTION_EXTRAPOLATION_FACTOR = 2.0    # 实际搜索窗口 = time_points.max() * factor

# ---- 负 OD 处理 ----
NEGATIVE_OD_FLOOR: Optional[float] = 0.0

# ---- Logistic 拟合 ----
N0_FLOOR = 1e3                             # 最低可检出 CFU/mL
R_UPPER = 2.0                              # r 上限 1/h

# ---- 交点标签布局 ----
INTERSECTION_LABEL_OFFSETS = {            # 单位：points，避免标签压住曲线
    '1e1': (0, -55),
    '1e2': (0, -35),
    '1e3': (0, 30),
    '1e4': (0, 18),
}


# =============================================================================
# 日志 / Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# =============================================================================
# 模型 / Model
# =============================================================================
def logistic(t, K, N0, r):
    """三参数 Logistic 生长模型。

    N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))
    """
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))


def r_squared(y_true, y_pred):
    """决定系数 R²,对零方差情形返回 1.0 避免除零。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - ss_res / ss_tot


def find_intersection_time(K, N0, r, target,
                           t_min: float = INTERSECTION_T_MIN,
                           t_max: float = INTERSECTION_T_MAX_DEFAULT
                           ) -> Optional[float]:
    """求 logistic 曲线穿过 `target` 的时间(解析初值 + fsolve)。"""
    def equation(t):
        return logistic(t, K, N0, r) - target

    if not (min(N0, K) < target < max(N0, K)):
        return None
    if r <= 0 or N0 <= 0 or K <= 0:
        return None

    try:
        term = (N0 * (K - target)) / ((K - N0) * target)
        if term <= 0:
            return None
        t_guess = -np.log(term) / r
        t_solution = fsolve(equation, t_guess)[0]
        if t_min <= t_solution <= t_max and \
                abs(equation(t_solution)) < INTERSECTION_TOLERANCE:
            return float(t_solution)
    except (ValueError, RuntimeError):
        return None
    return None


# =============================================================================
# 数据容器 / Data container
# =============================================================================
@dataclass
class SampleFit:
    """单样品拟合结果容器。"""
    name: str
    mean_cfu: Optional[np.ndarray] = None
    std_cfu: Optional[np.ndarray] = None
    raw_cfu: Optional[np.ndarray] = None
    K: Optional[float] = None
    N0: Optional[float] = None
    r: Optional[float] = None
    K_se: Optional[float] = None
    N0_se: Optional[float] = None
    r_se: Optional[float] = None
    R_squared: Optional[float] = None
    intersection_time: Optional[float] = None


# =============================================================================
# 样式 / Style
# =============================================================================
def setup_cns_style():
    """CNS 风格 matplotlib 全局设置(含中文字体回退)。"""
    import matplotlib.font_manager as fm
    try:
        fm._load_fontmanager(try_read_cache=False)
    except AttributeError:
        pass

    rcParams['font.sans-serif'] = [
        'Microsoft YaHei',   # Windows 主流
        'SimHei',            # Windows 简体
        'PingFang SC',       # macOS
        'Hiragino Sans GB',
        'STHeiti',
        'Noto Sans CJK SC',  # Linux
        'WenQuanYi Micro Hei',
        'Arial',             # 西文回退
    ]
    rcParams['font.family'] = 'sans-serif'
    rcParams['axes.unicode_minus'] = False

    rcParams['font.size'] = 14
    rcParams['axes.linewidth'] = 1.5
    rcParams['xtick.major.size'] = 6
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.size'] = 6
    rcParams['ytick.major.width'] = 1.2
    rcParams['legend.frameon'] = False
    rcParams['axes.grid'] = False
    rcParams['savefig.transparent'] = True
    rcParams['savefig.bbox'] = None       # 防止外部配置自动裁边改变宽高比
    rcParams['figure.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.spines.top'] = True
    rcParams['axes.spines.right'] = True


# =============================================================================
# 绘图 / Plotting
# =============================================================================
def plot_growth_CNS(time_points, sample_fits: List[SampleFit],
                    target_y: float, output_file: str, *,
                    show: bool = True) -> None:
    """CNS 风格 Logistic S 形生长曲线图（线性纵轴）。

    Parameters
    ----------
    time_points : array-like
        采样时间 (h)。
    sample_fits : list[SampleFit]
        各样品的拟合结果。
    target_y : float
        用于交点标记的目标 CFU/mL，不绘制水平阈值线。
    output_file : str
        图片保存路径。
    show : bool
        是否显示交互窗口；批量导出时可设为 False。
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    t_fit = np.linspace(0, time_points.max() + X_AXIS_PADDING, 1000)

    for idx, fit in enumerate(sample_fits):
        color = COLOR_MAP[idx % len(COLOR_MAP)]
        dark_color = darken_color(color)
        y_mean = fit.mean_cfu

        # 绘制所有单个数据点（深色标记，与OD深色点版本一致）
        if fit.raw_cfu is not None:
            raw = fit.raw_cfu
            for rep in range(raw.shape[0]):
                ax.scatter(time_points, raw[rep], color=dark_color, s=40, alpha=0.6, zorder=2)

        # 数据点 + 误差棒
        ax.errorbar(
            time_points, y_mean, yerr=fit.std_cfu,
            fmt='o', ms=5, lw=1.5, capsize=3,
            color=color, label=fit.name,
            markeredgecolor='black', markeredgewidth=0.4,
        )

        # 拟合曲线
        if fit.K is not None:
            y_fit = logistic(t_fit, fit.K, fit.N0, fit.r)
            ax.plot(t_fit, y_fit, '-', lw=2, color=color)

            # 交点 marker
            if fit.intersection_time is not None:
                ax.plot(fit.intersection_time, target_y, 'v',
                        color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.4)

        # 交点时间标签：按显示距离错开，适配线性纵轴
        if fit.K is not None and fit.intersection_time is not None:
            label_offset = INTERSECTION_LABEL_OFFSETS.get(fit.name, (0, 20))
            ax.annotate(
                f'{fit.intersection_time:.1f} h',
                xy=(fit.intersection_time, target_y),
                xytext=label_offset, textcoords='offset points',
                arrowprops=dict(
                    arrowstyle='-', color=color, lw=0.8, alpha=0.6,
                ),
                color=color, fontsize=11, ha='center',
                va='bottom' if label_offset[1] >= 0 else 'top',
                fontweight='bold',
            )

    # 轴范围
    ax.set_xscale('linear')
    ax.set_yscale('linear')
    ax.set_xlim(0, time_points.max() + X_AXIS_PADDING)
    y_data_max = max(
        (np.max(f.mean_cfu + (f.std_cfu if f.std_cfu is not None else 0))
         for f in sample_fits if f.mean_cfu is not None),
        default=target_y,
    )
    y_raw_max = max(
        (np.max(f.raw_cfu) for f in sample_fits if f.raw_cfu is not None),
        default=y_data_max,
    )
    K_max = max((f.K for f in sample_fits if f.K is not None),
                default=y_data_max)
    ax.set_ylim(Y_AXIS_LOWER_LIMIT,
                max(y_data_max, y_raw_max, K_max, target_y) * Y_AXIS_UPPER_FACTOR)
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.get_offset_text().set_fontsize(12)

    ax.set_xlabel('培养时间 (h)', fontsize=18, fontweight='bold')
    ax.set_ylabel('细菌浓度 (CFU/mL)', fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)

    # 图例(左上角,两列,无边框)
    handles, labels = ax.get_legend_handles_labels()
    labels_cn = [NAME_DISPLAY_CN.get(lab, lab) for lab in labels]
    ax.legend(handles, labels_cn, loc='upper left', fontsize=12, ncol=2)

    fig.tight_layout()
    # tight_layout 只调整画布内边距；bbox_inches='tight' 会裁剪画布，不能使用。
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(output_file, dpi=EXPORT_DPI, bbox_inches=None,
                    transparent=True)
    log.info("Figure saved to %s", os.path.abspath(output_file))
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Excel 导出 / Export
# =============================================================================
def export_results_to_excel(sample_fits: List[SampleFit],
                            time_points: np.ndarray,
                            fit_predictions: Dict[str, np.ndarray],
                            output_excel: str) -> None:
    """多 sheet 导出拟合结果。

    Sheets
    ------
    Parameters   K、N0、r 及标准误、R²、交点时间
    Raw_means    各时间点平均 CFU/mL
    Raw_stds     各时间点样本 SD(ddof=1)
    Predictions  拟合曲线在稠密网格上的预测值
    """
    rows = []
    for fit in sample_fits:
        rows.append({
            'Sample': fit.name,
            'K_CFU_per_mL': fit.K,
            'K_SE': fit.K_se,
            'N0_CFU_per_mL': fit.N0,
            'N0_SE': fit.N0_se,
            'r_per_h': fit.r,
            'r_SE': fit.r_se,
            'R_squared': fit.R_squared,
            'Intersection_h': fit.intersection_time,
        })
    df_params = pd.DataFrame(rows)

    df_means = pd.DataFrame(
        {fit.name: fit.mean_cfu for fit in sample_fits},
        index=time_points,
    )
    df_means.index.name = 'Time_h'

    df_stds = pd.DataFrame(
        {fit.name: fit.std_cfu for fit in sample_fits},
        index=time_points,
    )
    df_stds.index.name = 'Time_h'

    df_pred = pd.DataFrame(fit_predictions) if fit_predictions else pd.DataFrame()

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_params.to_excel(writer, sheet_name='Parameters', index=False)
        df_means.to_excel(writer, sheet_name='Raw_means')
        df_stds.to_excel(writer, sheet_name='Raw_stds')
        if not df_pred.empty:
            df_pred.to_excel(writer, sheet_name='Predictions', index=False)

    log.info("Results exported to %s", os.path.abspath(output_excel))


# =============================================================================
# 主流程 / Main pipeline
# =============================================================================
def analyze_growth(data: Dict[str, np.ndarray],
                   sample_names: List[str],
                   time_points: np.ndarray,
                   target_y: float,
                   conversion_factor: float,
                   output_file: str,
                   output_excel: str, *,
                   show: bool = True) -> List[SampleFit]:
    """完整生长曲线分析流水线。"""
    # --- 1. 预处理(不修改入参) -----------------------------
    data_proc: Dict[str, np.ndarray] = {}
    for name, arr in data.items():
        arr = np.asarray(arr, dtype=float)
        if NEGATIVE_OD_FLOOR is not None:
            arr = np.maximum(arr, NEGATIVE_OD_FLOOR)
        data_proc[name] = arr * conversion_factor

    sample_fits: List[SampleFit] = []
    for name in sample_names:
        arr = data_proc[name]
        sample_fits.append(SampleFit(
            name=name,
            mean_cfu=np.mean(arr, axis=0),
            std_cfu=np.std(arr, axis=0, ddof=1),
            raw_cfu=arr,
        ))

    # --- 2. 拟合(数据驱动 bounds) -----------------------------
    t_max_search = time_points.max() * INTERSECTION_EXTRAPOLATION_FACTOR
    for fit in sample_fits:
        y = fit.mean_cfu
        y_min = max(y.min(), 1.0)
        y_max = max(y.max(), y_min * 10)
        y_early = max(y[:3].mean(), 1.0)

        lower = [y_max, N0_FLOOR, 0.01]
        upper = [y_max * 5, max(2 * y_early, y_max), R_UPPER]
        p0 = [y_max * 2, max(y_early, N0_FLOOR), 0.3]

        try:
            popt, pcov = curve_fit(
                logistic, time_points, y,
                p0=p0, bounds=(lower, upper), maxfev=10000,
            )
            perr = (np.sqrt(np.diag(pcov))
                    if pcov is not None else (None, None, None))
            fit.K, fit.N0, fit.r = popt
            fit.K_se, fit.N0_se, fit.r_se = perr
            fit.R_squared = r_squared(y, logistic(time_points, *popt))

            if fit.K > fit.N0:
                fit.intersection_time = find_intersection_time(
                    *popt, target_y,
                    t_min=INTERSECTION_T_MIN,
                    t_max=t_max_search,
                )
            else:
                log.warning("Fit %s: K (%g) <= N0 (%g); skipping intersection.",
                            fit.name, fit.K, fit.N0)
                fit.intersection_time = None

            t_star = (f"{fit.intersection_time:.2f}h"
                      if fit.intersection_time is not None else "N/A")
            log.info(
                "Fit %s: K=%.3g±%.3g, N0=%.3g±%.3g, r=%.3g±%.3g, "
                "R²=%.4f, t*=%s",
                fit.name, fit.K, fit.K_se, fit.N0, fit.N0_se,
                fit.r, fit.r_se, fit.R_squared, t_star,
            )
        except (RuntimeError, ValueError) as e:
            log.warning("Fit failed for %s: %s", fit.name, e)

    # --- 3. 绘图 ---------------------------------------------------
    plot_growth_CNS(time_points, sample_fits, target_y, output_file, show=show)

    # --- 4. 预测 + 导出 --------------------------------------------
    t_grid = np.linspace(0, time_points.max() + X_AXIS_PADDING, 1000)
    pred_columns: Dict[str, np.ndarray] = {'t_h': t_grid}
    for fit in sample_fits:
        if fit.K is not None:
            pred_columns[f'{fit.name}_pred_CFU_per_mL'] = logistic(
                t_grid, fit.K, fit.N0, fit.r
            )

    export_results_to_excel(sample_fits, time_points, pred_columns,
                             output_excel)
    return sample_fits


# =============================================================================
# 示例用法 / Sample usage
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出 16:9 高清细菌生长曲线。')
    parser.add_argument('--no-show', action='store_true', help='仅保存文件，不显示图窗')
    args = parser.parse_args()
    setup_cns_style()

    data = {
        '1e1': np.array([
            [-0.013, -0.006, 0, -0.004, -0.004, -0.002, 0.004, 0.046, 0.366],
            [-0.018, -0.007, -0.003, -0.006, -0.005, -0.003, 0.011, 0.076, 0.436],
            [-0.013, -0.002, -0.005, 0, 0.001, 0.009, 0.031, 0.107, 0.378],
            [-0.018, 0, 0.001, -0.002, 0, 0.006, 0.02, 0.111, 0.386]]),
        '1e2': np.array([
            [-0.023, -0.008, -0.005, -0.002, 0.017, 0.118, 0.101, 0.183, 0.246],
            [-0.019, -0.007, -0.006, 0.004, 0.019, 0.095, 0.164, 0.212, 0.364],
            [-0.021, -0.007, -0.002, 0.005, 0.013, 0.112, 0.221, 0.167, 0.352],
            [-0.022, 0.001, -0.002, -0.002, 0.018, 0.087, 0.237, 0.26, 0.279]]),
        '1e3': np.array([
            [-0.02, -0.001, 0.013, 0.112, 0.111, 0.213, 0.236, 0.3, 0.329],
            [-0.016, -0.002, 0.009, 0.069, 0.069, 0.292, 0.281, 0.296, 0.326],
            [-0.015, -0.004, 0.014, 0.089, 0.095, 0.231, 0.31, 0.325, 0.316],
            [-0.009, 0, 0.015, 0.08, 0.107, 0.315, 0.26, 0.252, 0.304]]),
        '1e4': np.array([
            [0.027, 0.047, 0.086, 0.135, 0.248, 0.362, 0.429, 0.442, 0.443],
            [0.023, 0.043, 0.074, 0.117, 0.209, 0.331, 0.372, 0.47, 0.43],
            [0.021, 0.046, 0.082, 0.123, 0.227, 0.345, 0.4, 0.491, 0.444],
            [0.032, 0.048, 0.083, 0.136, 0.239, 0.36, 0.41, 0.43, 0.43]]),
    }
    sample_names = ['1e1', '1e2', '1e3', '1e4']
    time_points = np.array([0, 2, 4, 6, 8, 10, 12, 14, 24])
    target_y = 2e8
    conversion_factor = 8e8
    output_dir = Path(__file__).resolve().parent
    output_file = output_dir / "bacterial_growth_20260924_16-9.png"
    output_excel = output_dir / "growth_curve_fitting_results_20260924.xlsx"

    try:
        analyze_growth(
            data=data,
            sample_names=sample_names,
            time_points=time_points,
            target_y=target_y,
            conversion_factor=conversion_factor,
            output_file=output_file,
            output_excel=output_excel,
            show=not args.no_show,
        )
    except Exception:
        log.exception("Analysis pipeline failed.")
        raise
