# -*- coding: utf-8 -*-
"""
Battery Dataset EDA + XGBoost/SHAP (多张大图，单图合并多个子图)

运行方式（示例）：
    python eda_battery_analysis.py --csv /mnt/data/battery_dataset.csv --outdir /mnt/data/eda_outputs

说明：
- 本脚本严格复现你给出的“数据预处理/特征构造”，并在其基础上做探索性分析（EDA）。
- 输出：多张“大图”（每张图包含多子图），覆盖：
  1) 成对关系图（多对子图合并）
  2) 相关性热力图（Pearson/Spearman）
  3) 小提琴图（按类别/状态分组）
  4) 等高线图（二维分箱均值）
  5) 3D 曲面图（二维分箱均值）
  6) 3D 瀑布图（按分组偏移的 3D 线框）
  7) XGBoost + SHAP：融合“蜂群图 + 特征重要性条形图”的同图双 X 轴展示
"""

from __future__ import annotations

import os
import math
import argparse
import typing
import warnings
import xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# 0) 统一配色（来自你给的图片）
# =========================
PALETTE = [
    "#91B5DC",  # blue
    "#8DC9AC",  # green
    "#DAACB8",  # mauve
    "#F2D1E1",  # pink
    "#F9EEAE",  # pale yellow
    "#BDDECF",  # mint
    "#C7DDEC",  # light blue
]

CMAP_SOFT = LinearSegmentedColormap.from_list("soft_palette", PALETTE, N=256)

def set_mpl_style() -> None:
    """尽量保证“美观且一致”的默认风格（不依赖 seaborn 风格）。"""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })

# =========================
# 1) 你给出的预处理（原样嵌入）
# =========================

def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def compute_opt_temperature(temp_c: pd.Series) -> np.ndarray:
    """
    计算题设中的 T(t)（“最佳温度/有效温度”）——按你的要求“需要通过算法计算得出”。

    预处理步骤（已标注）：
    1) 强制转数值，非法转NaN；
    2) 线性插值补齐缺失（再用中位数兜底）；
    3) Winsorize 去极值：按 1%~99% 分位裁剪（防止传感器尖峰）；
    4) 平滑：rolling median(5) + EWMA(span=10) 降噪，得到 T_opt(t)。
    """
    x = pd.to_numeric(temp_c, errors="coerce")

    # (1)(2) 缺失处理：插值 + 中位数兜底
    x = x.interpolate(limit_direction="both")
    x = x.fillna(x.median())

    # (3) 去极值：按分位裁剪
    lo = float(x.quantile(0.01))
    hi = float(x.quantile(0.99))
    x = x.clip(lower=lo, upper=hi)

    # (4) 平滑：滚动中位数 + 指数滑动平均
    x_med = x.rolling(window=5, center=True, min_periods=1).median()
    x_ewm = x_med.ewm(span=10, adjust=False).mean()

    return x_ewm.to_numpy(dtype=float)

def build_model_df_from_battery_dataset(
    path: str,
    choose_device: typing.Optional[str] = None
) -> typing.Tuple[pd.DataFrame, np.ndarray]:
    """
    输入：battery_dataset.csv
    输出：
      df_model：包含模型需要的特征列
        time_s, soc, temp_c(原始), temp_opt_c(算法得到的T(t)),
        screen_on, brightness, cpu_util, net_mode, gps_on, bg_on, bg_intensity
      P_obs_W：观测总功耗（W）
    """
    raw = pd.read_csv(path)

    # ---- (预处理1) 解析时间戳 ----
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).copy()

    # ---- (预处理2) 多设备处理：选一个 device_id 来保持原主流程单序列 ----
    if "device_id" in raw.columns:
        if choose_device is None:
            vc = raw["device_id"].astype(str).value_counts()
            choose_device = str(vc.index[0])  # 样本最多的设备
        raw = raw[raw["device_id"].astype(str) == str(choose_device)].copy()

    # ---- (预处理1续) 排序，确保时序一致 ----
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    # ---- 计算 time_s：相对秒 ----
    t0 = raw["timestamp"].min()
    time_s = (raw["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)

    # ---- 构造模型特征表 ----
    model = pd.DataFrame()
    model["time_s"] = time_s

    # SOC：强制数值+裁剪
    soc = pd.to_numeric(raw["SOC"], errors="coerce").interpolate(limit_direction="both")
    soc = soc.fillna(soc.median())
    model["soc"] = clip01(soc.to_numpy(dtype=float))

    # 温度：原始 temp_c 与 算法得到的 temp_opt_c (=题设 T(t))
    temp_c = pd.to_numeric(raw["temperature"], errors="coerce").interpolate(limit_direction="both")
    temp_c = temp_c.fillna(temp_c.median())
    model["temp_c"] = temp_c.to_numpy(dtype=float)
    model["temp_opt_c"] = compute_opt_temperature(temp_c)

    # 亮度：强制数值+裁剪
    b = pd.to_numeric(raw["brightness"], errors="coerce").interpolate(limit_direction="both")
    b = b.fillna(b.median())
    model["brightness"] = clip01(b.to_numpy(dtype=float))

    # screen_on：用亮度阈值推断（经验阈值 0.12）
    model["screen_on"] = (model["brightness"].to_numpy() >= 0.12).astype(float)

    # CPU负载：强制数值+裁剪
    u = pd.to_numeric(raw["cpu_load"], errors="coerce").interpolate(limit_direction="both")
    u = u.fillna(u.median())
    model["cpu_util"] = clip01(u.to_numpy(dtype=float))

    # 网络模式：统一小写 + 兜底
    nm = raw["network_type"].astype(str).str.strip().str.lower()
    nm = nm.where(nm.isin(["wifi", "4g", "5g"]), other="wifi")
    model["net_mode"] = nm.to_numpy()

    # GPS：bool/字符串 -> 0/1
    gps = raw["gps_enabled"]
    if gps.dtype == object:
        gps = gps.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    model["gps_on"] = gps.astype(float).to_numpy()

    # 后台：强度=后台app数归一化；bg_on=屏幕灭 & 后台app>0
    nba = pd.to_numeric(raw["num_background_apps"], errors="coerce").interpolate(limit_direction="both")
    nba = nba.fillna(nba.median()).clip(lower=0.0)
    nba_np = nba.to_numpy(dtype=float)
    max_apps = float(np.max(nba_np)) if np.max(nba_np) > 0 else 1.0
    model["bg_intensity"] = clip01(nba_np / max_apps)
    model["bg_on"] = (((model["screen_on"].to_numpy() < 0.5) & (nba_np > 0.0))).astype(float)

    # 观测功耗：mW->W
    p_mw = pd.to_numeric(raw["power_consumption_mw"], errors="coerce").interpolate(limit_direction="both")
    p_mw = p_mw.fillna(p_mw.median()).clip(lower=0.0)
    P_obs_W = p_mw.to_numpy(dtype=float) * 1e-3

    return model, P_obs_W


# =========================
# 2) 常用小工具
# =========================

def ensure_outdir(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def savefig(fig: plt.Figure, outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path

def zscore(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    mu = np.nanmean(a)
    sd = np.nanstd(a) + 1e-12
    return (a - mu) / sd

def robust_sample(df: pd.DataFrame, n: int = 6000, seed: int = 7) -> pd.DataFrame:
    """用于大型散点图的子采样（让图更清晰且速度更稳）。"""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def binned_mean_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bins: int = 40,
    xlim: typing.Optional[typing.Tuple[float, float]] = None,
    ylim: typing.Optional[typing.Tuple[float, float]] = None,
) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    二维分箱后求 z 的均值：用于等高线/曲面图。
    返回：Xc, Yc, Zmean，其中 Zmean 为空箱用 NaN。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    if xlim is None:
        xlim = (np.nanquantile(x, 0.01), np.nanquantile(x, 0.99))
    if ylim is None:
        ylim = (np.nanquantile(y, 0.01), np.nanquantile(y, 0.99))

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]

    xedges = np.linspace(xlim[0], xlim[1], bins + 1)
    yedges = np.linspace(ylim[0], ylim[1], bins + 1)

    # 累积和 & 计数
    sum_z, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=z)
    cnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])

    with np.errstate(invalid="ignore", divide="ignore"):
        zmean = sum_z / cnt
    zmean[cnt == 0] = np.nan

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    return Xc, Yc, zmean

# =========================
# 3) EDA 图表 1：成对关系图（多对子图合并）
# =========================

def fig_pairwise_relationships(df: pd.DataFrame, outdir: str) -> str:
    """
    大图：多个成对散点 + 局部平滑趋势（按 net_mode 上色）
    """
    d = robust_sample(df, n=9000, seed=7)

    pairs = [
        ("time_s", "soc"),
        ("time_s", "brightness"),
        ("time_s", "cpu_util"),
        ("time_s", "temp_opt_c"),
        ("time_s", "bg_intensity"),
        ("time_s", "P_obs_W")
    ]
    n = len(pairs)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Pairwise Relationships (colored by net_mode)", y=1.02)

    # 手工映射 net_mode -> 颜色
    nm_order = ["wifi", "4g", "5g"]
    nm_color = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(nm_order)}

    for i, (xcol, ycol) in enumerate(pairs, start=1):
        ax = fig.add_subplot(nrows, ncols, i)

        for nm in nm_order:
            dd = d[d["net_mode"] == nm]
            ax.scatter(
                dd[xcol], dd[ycol],
                s=8, alpha=0.35, linewidths=0,
                label=nm if i == 1 else None,
                color=nm_color[nm],
            )

        # 简易滚动分箱趋势：按 x 分位点分箱后画均值线（稳健、可复现）
        x = d[xcol].to_numpy()
        y = d[ycol].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if len(x) > 50:
            qs = np.linspace(0.02, 0.98, 25)
            edges = np.quantile(x, qs)
            idx = np.digitize(x, edges)
            xs, ys = [], []
            for k in range(1, len(edges)):
                mk = idx == k
                if np.sum(mk) >= 30:
                    xs.append(np.median(x[mk]))
                    ys.append(np.median(y[mk]))
            if len(xs) >= 3:
                ax.plot(xs, ys, lw=2.2, alpha=0.9, color=PALETTE[-1])

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{ycol} vs {xcol}")

    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    return savefig(fig, outdir, "01_pairwise_relationships.png")

# =========================
# 4) EDA 图表 2：相关性热力图（Pearson + Spearman）
# =========================

def fig_correlation_heatmaps(df: pd.DataFrame, outdir: str) -> str:
    num_cols = ["soc", "temp_c", "temp_opt_c", "screen_on", "brightness", "cpu_util",
                "gps_on", "bg_on", "bg_intensity", "time_s", "P_obs_W"]
    d = df[num_cols].copy()

    pear = d.corr(method="pearson")
    spear = d.corr(method="spearman")

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Correlation Heatmaps (Pearson / Spearman)", y=1.02)

    for j, (M, title) in enumerate([(pear, "Pearson"), (spear, "Spearman")], start=1):
        ax = fig.add_subplot(1, 2, j)
        im = ax.imshow(M.to_numpy(), cmap=CMAP_SOFT, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(M.columns)))
        ax.set_yticks(range(len(M.index)))
        ax.set_xticklabels(M.columns, rotation=45, ha="right")
        ax.set_yticklabels(M.index)

        # 数值标注（避免太密：只标注|corr|>=0.35）
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                v = float(M.iat[r, c])
                if abs(v) >= 0.35:
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("corr", rotation=90)

    return savefig(fig, outdir, "02_correlation_heatmaps.png")

# =========================
# 5) EDA 图表 3：小提琴图（分组分布）
# =========================

def fig_violin_distributions(df: pd.DataFrame, outdir: str) -> str:
    """
    多个小提琴图合并：按 net_mode / screen_on / gps_on / bg_on 分组展示 P_obs_W 的分布
    """
    try:
        import seaborn as sns
    except Exception as e:
        raise RuntimeError("需要 seaborn 来画小提琴图。请先安装：pip install seaborn") from e

    # 用柔和调色板（保证与图片颜色一致）
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "white"})
    sns.set_palette(PALETTE)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Violin Plots: Power Distribution by Modes/States", y=1.02)

    # 1) net_mode
    ax1 = fig.add_subplot(2, 2, 1)
    sns.violinplot(data=df, x="net_mode", y="P_obs_W", cut=0, inner="quartile", ax=ax1)
    ax1.set_title("P_obs_W by net_mode")
    ax1.set_xlabel("net_mode")
    ax1.set_ylabel("P_obs_W (W)")

    # 2) screen_on
    ax2 = fig.add_subplot(2, 2, 2)
    df2 = df.copy()
    df2["screen_on"] = df2["screen_on"].map({0.0: "off", 1.0: "on"})
    sns.violinplot(data=df2, x="screen_on", y="P_obs_W", cut=0, inner="quartile", ax=ax2)
    ax2.set_title("P_obs_W by screen_on")
    ax2.set_xlabel("screen_on")
    ax2.set_ylabel("P_obs_W (W)")

    # 3) gps_on
    ax3 = fig.add_subplot(2, 2, 3)
    df3 = df.copy()
    df3["gps_on"] = df3["gps_on"].map({0.0: "off", 1.0: "on"})
    sns.violinplot(data=df3, x="gps_on", y="P_obs_W", cut=0, inner="quartile", ax=ax3)
    ax3.set_title("P_obs_W by gps_on")
    ax3.set_xlabel("gps_on")
    ax3.set_ylabel("P_obs_W (W)")

    # 4) bg_on
    ax4 = fig.add_subplot(2, 2, 4)
    df4 = df.copy()
    df4["bg_on"] = df4["bg_on"].map({0.0: "off", 1.0: "on"})
    sns.violinplot(data=df4, x="bg_on", y="P_obs_W", cut=0, inner="quartile", ax=ax4)
    ax4.set_title("P_obs_W by bg_on")
    ax4.set_xlabel("bg_on")
    ax4.set_ylabel("P_obs_W (W)")

    return savefig(fig, outdir, "03_violin_distributions.png")

# =========================
# 6) EDA 图表 4：等高线图（二维分箱均值）
# =========================

def fig_contours(df: pd.DataFrame, outdir: str) -> str:
    """
    等高线（或填色等高线）图：展示两个特征与功耗的关系（通过 2D 分箱均值平滑）
    """
    triplets = [
        ("brightness", "cpu_util", "P_obs_W"),
        ("temp_opt_c", "cpu_util", "P_obs_W"),
        ("temp_opt_c", "brightness", "P_obs_W"),
        ("soc", "cpu_util", "P_obs_W"),
    ]

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Contour Maps (binned mean of P_obs_W)", y=1.02)

    for i, (xcol, ycol, zcol) in enumerate(triplets, start=1):
        ax = fig.add_subplot(2, 2, i)
        Xc, Yc, Zm = binned_mean_2d(df[xcol].to_numpy(), df[ycol].to_numpy(), df[zcol].to_numpy(), bins=45)
        # transpose：imshow/contour 常用 (Y, X)，这里直接 contourf 用 Xc,Yc 的 mesh
        levels = 18
        cf = ax.contourf(Xc, Yc, Zm, levels=levels, cmap=CMAP_SOFT)
        cs = ax.contour(Xc, Yc, Zm, levels=8, colors="k", linewidths=0.5, alpha=0.35)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{zcol} mean over ({xcol}, {ycol}) bins")
        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(zcol)

    return savefig(fig, outdir, "04_contour_maps.png")

# =========================
# 7) EDA 图表 5：3D 曲面图（二维分箱均值）
# =========================

def fig_3d_surfaces(df: pd.DataFrame, outdir: str) -> str:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    surfaces = [
        ("brightness", "cpu_util", "P_obs_W"),
        ("temp_opt_c", "cpu_util", "P_obs_W"),
        ("temp_opt_c", "brightness", "P_obs_W"),
    ]

    fig = plt.figure(figsize=(18, 6))
    fig.suptitle("3D Surface Plots (binned mean of P_obs_W)", y=1.02)

    for i, (xcol, ycol, zcol) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        Xc, Yc, Zm = binned_mean_2d(df[xcol].to_numpy(), df[ycol].to_numpy(), df[zcol].to_numpy(), bins=35)
        # Zm 中 NaN 会影响曲面，简单填充：用全局中位数
        zfill = np.nanmedian(df[zcol].to_numpy())
        Zplot = np.where(np.isfinite(Zm), Zm, zfill)

        ax.plot_surface(Xc, Yc, Zplot, cmap=CMAP_SOFT, rstride=1, cstride=1, linewidth=0.2, antialiased=True, alpha=0.95)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_zlabel(zcol)
        ax.set_title(f"{zcol} surface over ({xcol}, {ycol})")

        # 观感优化：轻微调整视角
        ax.view_init(elev=25, azim=235)

    return savefig(fig, outdir, "05_3d_surfaces.png")

# =========================
# 8) EDA 图表 6：3D 瀑布图（Waterfall）
# =========================

def fig_3d_waterfall(df: pd.DataFrame, outdir: str) -> str:
    """
    3D 瀑布图：按 net_mode 分组，展示 P_obs_W 随 SOC 的变化曲线，并在 y 方向做分组偏移。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("3D Waterfall: P_obs_W vs SOC (grouped by net_mode)", y=0.98)

    groups = ["wifi", "4g", "5g"]
    y_offsets = {g: i for i, g in enumerate(groups)}

    for i, g in enumerate(groups):
        d = df[df["net_mode"] == g].copy()
        if len(d) < 100:
            continue

        # 分箱平滑：按 SOC 分 60 个箱取中位数
        soc = d["soc"].to_numpy()
        p = d["P_obs_W"].to_numpy()
        m = np.isfinite(soc) & np.isfinite(p)
        soc, p = soc[m], p[m]

        edges = np.linspace(np.nanquantile(soc, 0.01), np.nanquantile(soc, 0.99), 61)
        idx = np.digitize(soc, edges)
        xs, zs = [], []
        for k in range(1, len(edges)):
            mk = idx == k
            if np.sum(mk) >= 20:
                xs.append(np.median(soc[mk]))
                zs.append(np.median(p[mk]))
        if len(xs) < 5:
            continue

        y = np.full(len(xs), y_offsets[g], dtype=float)
        ax.plot(xs, y, zs, lw=2.5, alpha=0.95, label=g)

        # 在曲线下方补一个“阴影填充”，增强瀑布层次感（简化版）
        ax.plot(xs, y, np.zeros_like(zs), lw=0.8, alpha=0.25)

    ax.set_xlabel("soc")
    ax.set_ylabel("net_mode (offset)")
    ax.set_zlabel("P_obs_W (W)")
    ax.set_yticks([y_offsets[g] for g in groups])
    ax.set_yticklabels(groups)
    ax.view_init(elev=25, azim=230)
    ax.legend(frameon=False, loc="upper left")

    return savefig(fig, outdir, "06_3d_waterfall.png")

# =========================
# 9) XGBoost + SHAP（融合蜂群 + 重要性条形图，双 X 轴）
# =========================

def _fit_xgb_and_shap(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = 7
):
    """
    尝试用 xgboost + shap；若缺少依赖则抛出明确错误信息。
    """
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError("需要 xgboost。请先安装：pip install xgboost") from e

    try:
        import shap
    except Exception as e:
        raise RuntimeError("需要 shap。请先安装：pip install shap") from e

    # 训练/验证切分
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = xgb.XGBRegressor(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        objective="reg:squarederror",
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    # SHAP（TreeExplainer）
    explainer = shap.TreeExplainer(model)
    # 为了稳定和速度：对测试集再采样
    Xt = X_test
    if len(Xt) > 4000:
        Xt = Xt.sample(n=4000, random_state=seed)
    shap_values = explainer.shap_values(Xt)

    return model, explainer, Xt, shap_values, metrics

def fig_xgb_shap_dual_axis(df: pd.DataFrame, outdir: str) -> str:
    """
    单张大图：左侧为“蜂群散点”（SHAP 值分布 + 影响方向 + 特征取值分布），
    右侧/双 X 轴为“平均|SHAP|”条形重要性（整体贡献大小）。
    """
    # ----------- 1) 特征工程（net_mode one-hot）-----------
    feat_cols = ["soc", "temp_opt_c", "screen_on", "brightness", "cpu_util", "gps_on", "bg_on", "bg_intensity"]
    X_num = df[feat_cols].copy()

    X_cat = pd.get_dummies(df["net_mode"], prefix="net", drop_first=False)
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["P_obs_W"].to_numpy(dtype=float)

    model, explainer, Xt, shap_vals, metrics = _fit_xgb_and_shap(X, y, seed=7)

    # ----------- 2) 选 Top-K 特征 -----------
    # shap_vals shape: (n_samples, n_features)
    abs_mean = np.mean(np.abs(shap_vals), axis=0)
    feat_names = np.array(X.columns)
    order = np.argsort(abs_mean)[::-1]
    top_k = min(12, len(order))
    top_idx = order[:top_k]

    # 取 Top 特征子集
    shap_top = shap_vals[:, top_idx]
    X_top = Xt.iloc[:, top_idx].copy()
    names_top = feat_names[top_idx]
    imp_top = abs_mean[top_idx]

    # ----------- 3) 绘图布局 -----------
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    fig.suptitle(
        f"XGBoost + SHAP (Dual X-Axis Beeswarm + Importance) | "
        f"R²={metrics['r2']:.3f}, MAE={metrics['mae']:.4f}",
        y=1.02
    )

    # y 轴：特征名（从上到下重要性递减）
    y_positions = np.arange(top_k)

    # 为了让重要性从上到下：把顺序翻转（顶部最重要）
    y_positions = y_positions[::-1]
    names_plot = list(names_top[::-1])
    shap_plot = shap_top[:, ::-1]
    X_plot = X_top.iloc[:, ::-1]
    imp_plot = imp_top[::-1]

    # 颜色：用“特征值大小”映射到柔和 colormap
    # 对每个特征独立归一化（更能体现分布）
    rng = np.random.default_rng(7)

    for j in range(top_k):
        vals = X_plot.iloc[:, j].to_numpy(dtype=float)
        sv = shap_plot[:, j].astype(float)

        # jitter：避免点完全重叠
        jitter = (rng.random(len(sv)) - 0.5) * 0.55
        yy = np.full(len(sv), y_positions[j], dtype=float) + jitter

        # 归一化特征值用于上色；若是 one-hot 则直接 0/1
        vmin = np.nanquantile(vals, 0.02)
        vmax = np.nanquantile(vals, 0.98)
        denom = (vmax - vmin) if (vmax - vmin) > 1e-12 else 1.0
        c = np.clip((vals - vmin) / denom, 0, 1)

        ax.scatter(
            sv, yy,
            s=12, alpha=0.45, linewidths=0,
            c=c, cmap=CMAP_SOFT,
        )

        # 0 参考线（影响方向）
        ax.axvline(0, lw=1.0, alpha=0.35, color="k")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(names_plot)
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_ylabel("Features (top importance)")

    # ----------- 4) 双 X 轴：平均|SHAP|重要性条形图 -----------
    ax2 = ax.twiny()
    # 把 bar 放在“特征行”的右侧：用负宽度/透明度避免遮挡
    # 这里使用 ax2 的数据坐标：直接画水平条形
    ax2.barh(
        y_positions,
        imp_plot,
        height=0.55,
        alpha=0.25,
        color=PALETTE[0],
        edgecolor="none"
    )
    ax2.set_xlabel("mean(|SHAP|) importance")

    # 统一 y 范围
    ax.set_ylim(-1, top_k)
    ax2.set_ylim(-1, top_k)

    # 颜色条（说明点颜色=特征值）
    sm = plt.cm.ScalarMappable(cmap=CMAP_SOFT)
    sm.set_array(np.linspace(0, 1, 10))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("feature value (low → high)")

    # 文本注释：top 特征的重要性数值
    for j in range(top_k):
        ax2.text(
            imp_plot[j],
            y_positions[j],
            f" {imp_plot[j]:.3f}",
            va="center",
            ha="left",
            fontsize=8,
            alpha=0.8
        )

    return savefig(fig, outdir, "07_xgb_shap_dual_axis.png")

# =========================
# 10) 主流程：读数据 -> 预处理 -> 画图 -> 保存
# =========================

def main():
    set_mpl_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="battery_dataset.csv", help="battery_dataset.csv 路径")
    ap.add_argument("--outdir", type=str, default="eda_outputs", help="输出目录")
    ap.add_argument("--device", type=str, default=None, help="可选：指定 device_id（多设备时）")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)

    df_model, P_obs_W = build_model_df_from_battery_dataset(args.csv, choose_device=args.device)
    df = df_model.copy()
    df["P_obs_W"] = P_obs_W

    # ---- 简单文本报告：保存成 CSV/MD，便于论文复现 ----
    report = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "net_mode_counts": df["net_mode"].value_counts(dropna=False).to_dict(),
        "P_obs_W_summary": df["P_obs_W"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
        "missing_ratio": (df.isna().mean()).to_dict(),
    }
    with open(os.path.join(outdir, "eda_report.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 依次生成大图 ----
    paths = []
    paths.append(fig_pairwise_relationships(df, outdir))
    paths.append(fig_correlation_heatmaps(df, outdir))
    paths.append(fig_violin_distributions(df, outdir))
    paths.append(fig_contours(df, outdir))
    paths.append(fig_3d_surfaces(df, outdir))
    paths.append(fig_3d_waterfall(df, outdir))

    # XGBoost + SHAP：若环境缺依赖，这里会给出明确错误
    try:
        paths.append(fig_xgb_shap_dual_axis(df, outdir))
    except Exception as e:
        # 给出“不会中断其它图”的处理：把错误写入文件
        with open(os.path.join(outdir, "xgb_shap_error.txt"), "w", encoding="utf-8") as f:
            f.write(str(e))

    # 保存一个索引文件，方便你快速检查输出
    with open(os.path.join(outdir, "outputs_index.txt"), "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")

    print("Done. Outputs saved to:", outdir)
    for p in paths:
        print(" -", p)

if __name__ == "__main__":
    main()
