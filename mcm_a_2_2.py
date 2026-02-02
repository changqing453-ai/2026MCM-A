# -*- coding: utf-8 -*-
"""
compare_variable_sensitivity.py

目标（彻底重构版）：
1) 功耗公式（predict_power_step / power_components_step）严格不改。
2) 不修改参数 pp（p0, ps0, ...），只修改“变量/控制量”（brightness, cpu_util, net_mode, gps_on, bg_on, bg_intensity...）。
3) 在多种典型使用情境（scenarios）下，对变量进行微扰与切换，输出敏感性分析图表：
   (1) Bar: Sensitivity of Total Power (ΔP_total)
   (2) Heatmap: Component Power Share
   (3) Stacked Bar: Power Component Breakdown
   (4) Ridgeline: Power Distribution under Perturbations

依赖：numpy, pandas, matplotlib
可选：如果存在 fitted_params.json，会用它作为参数 pp（但绝不改 pp）
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 输出目录与全局设置
# =========================
OUT_DIR = "figs_sensitivity"
SAVE_DPI = 300
os.makedirs(OUT_DIR, exist_ok=True)

# 海洋 + 森系配色（你可按喜好替换，但保持风格一致）
OCEAN_FOREST = {
    "ocean_deep":   "#0B3C5D",
    "ocean":        "#1C6E8C",
    "ocean_light":  "#74B3CE",
    "seafoam":      "#A7DCCB",
    "forest_deep":  "#1B4332",
    "forest":       "#2D6A4F",
    "forest_light": "#74C69D",
    "moss":         "#95D5B2",
    "sand":         "#E9D8A6",
    "stone":        "#CAD2C5",
    "ink":          "#1F2937",
    "grid":         "#9AA6B2",
}

# 分量颜色（海洋+森系）
COMP_COLOR = {
    "P_base":   OCEAN_FOREST["stone"],
    "P_scr":    OCEAN_FOREST["ocean_light"],
    "P_cpu":    OCEAN_FOREST["ocean"],
    "P_wifi":   OCEAN_FOREST["seafoam"],
    "P_4g":     OCEAN_FOREST["sand"],
    "P_5g":     OCEAN_FOREST["forest_light"],
    "P_gps":    OCEAN_FOREST["forest"],
    "P_bg_sw":  OCEAN_FOREST["moss"],
    "P_bg_int": OCEAN_FOREST["forest_deep"],
}

COMP_COLS = ["P_base", "P_scr", "P_cpu", "P_wifi", "P_4g", "P_5g", "P_gps", "P_bg_sw", "P_bg_int"]


def _save_fig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    return path


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


# =========================
# 1) 参数容器（只读，不改）
# =========================
@dataclass(frozen=True)
class PowerParams:
    p0: float
    ps0: float
    ps1: float
    pc0: float
    pc1: float
    gamma: float
    pwifi: float
    p4g: float
    p5g: float
    pgps: float
    pbg0: float
    pbg1: float


def load_params_from_json(path: str) -> PowerParams:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    keys = {k: float(d[k]) for k in PowerParams.__annotations__.keys() if k in d}
    missing = [k for k in PowerParams.__annotations__.keys() if k not in keys]
    if missing:
        raise ValueError(f"JSON 缺少字段：{missing}")
    return PowerParams(**keys)


# =========================
# 2) 功耗公式（严格不改）
# =========================
def predict_power_step(
    pp: PowerParams,
    area_ratio: float,
    screen_on: float,
    brightness: float,
    cpu_util: float,
    net_mode: str,
    gps_on: float,
    bg_on: float,
    bg_intensity: float,
) -> float:
    """
    严格按题设最终功率公式计算单步功耗（不允许改）：
      P = p0
        + Ar*I_scr*(ps0 + ps1*b)
        + I_cpu*(pc0 + pc1*u^gamma)
        + I_wifi*pwifi + I_4g*p4g + I_5g*p5g
        + I_gps*pgps
        + I_bg*pbg0
        + pbg1*a_bg
    """
    b = float(np.clip(brightness, 0.0, 1.0))
    u = float(np.clip(cpu_util, 0.0, 1.0))

    I_scr = 1.0 if float(screen_on) >= 0.5 else 0.0
    I_cpu = 1.0 if u > 1e-3 else 0.0
    I_gps = 1.0 if float(gps_on) >= 0.5 else 0.0
    I_bg  = 1.0 if float(bg_on)  >= 0.5 else 0.0

    nm = (net_mode or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g   = 1.0 if nm == "4g"   else 0.0
    I_5g   = 1.0 if nm == "5g"   else 0.0

    ug = u ** float(pp.gamma)

    P = float(pp.p0)
    P += float(area_ratio) * I_scr * (float(pp.ps0) + float(pp.ps1) * b)
    P += I_cpu * (float(pp.pc0) + float(pp.pc1) * ug)
    P += I_wifi * float(pp.pwifi) + I_4g * float(pp.p4g) + I_5g * float(pp.p5g)
    P += I_gps * float(pp.pgps)
    P += I_bg * float(pp.pbg0)
    # 与题设一致：强度项不额外乘 I_bg
    P += float(pp.pbg1) * float(np.clip(bg_intensity, 0.0, 1.0))

    return max(P, 0.0)


def power_components_step(
    pp: PowerParams,
    area_ratio: float,
    screen_on: float,
    brightness: float,
    cpu_util: float,
    net_mode: str,
    gps_on: float,
    bg_on: float,
    bg_intensity: float,
) -> Dict[str, float]:
    """同一套功耗公式下的分量拆解（用于占比/堆叠）。"""
    b = float(np.clip(brightness, 0.0, 1.0))
    u = float(np.clip(cpu_util, 0.0, 1.0))

    I_scr = 1.0 if float(screen_on) >= 0.5 else 0.0
    I_cpu = 1.0 if u > 1e-3 else 0.0
    I_gps = 1.0 if float(gps_on) >= 0.5 else 0.0
    I_bg  = 1.0 if float(bg_on)  >= 0.5 else 0.0

    nm = (net_mode or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g   = 1.0 if nm == "4g"   else 0.0
    I_5g   = 1.0 if nm == "5g"   else 0.0

    ug = u ** float(pp.gamma)

    comps = {}
    comps["P_base"]   = float(pp.p0)
    comps["P_scr"]    = float(area_ratio) * I_scr * (float(pp.ps0) + float(pp.ps1) * b)
    comps["P_cpu"]    = I_cpu * (float(pp.pc0) + float(pp.pc1) * ug)
    comps["P_wifi"]   = I_wifi * float(pp.pwifi)
    comps["P_4g"]     = I_4g * float(pp.p4g)
    comps["P_5g"]     = I_5g * float(pp.p5g)
    comps["P_gps"]    = I_gps * float(pp.pgps)
    comps["P_bg_sw"]  = I_bg * float(pp.pbg0)
    comps["P_bg_int"] = float(pp.pbg1) * float(np.clip(bg_intensity, 0.0, 1.0))
    comps["P_total"]  = sum(comps.values())
    return comps


# =========================
# 3) 典型情境输入（你可替换为论文10情境）
# =========================
def default_scenarios() -> pd.DataFrame:
    rows = [
        ("video_wifi",              0.95, 1, 0.70, 0.35, "wifi", 0, 1, 0.30),
        ("shortvideo_highbright",   0.95, 1, 0.95, 0.45, "wifi", 0, 1, 0.35),
        ("browsing_lowbright",      0.95, 1, 0.35, 0.20, "wifi", 0, 1, 0.20),
        ("social_wifi",             0.95, 1, 0.55, 0.25, "wifi", 0, 1, 0.25),
        ("gaming_wifi",             0.95, 1, 0.65, 0.85, "wifi", 0, 1, 0.40),
        ("videocall_wifi",          0.95, 1, 0.60, 0.55, "wifi", 0, 1, 0.35),
        ("navigation_5g_gps",       0.95, 1, 0.70, 0.30, "5g",   1, 1, 0.30),
        ("weaknet_4g_burst",        0.95, 1, 0.60, 0.35, "4g",   0, 1, 0.60),
        ("music_screenoff",         0.95, 0, 0.00, 0.05, "wifi", 0, 1, 0.30),
        ("idle_bg_medium",          0.95, 0, 0.00, 0.01, "none", 0, 1, 0.50),
    ]
    df = pd.DataFrame(rows, columns=[
        "scenario", "area_ratio", "screen_on", "brightness", "cpu_util",
        "net_mode", "gps_on", "bg_on", "bg_intensity"
    ])
    return df


# =========================
# 4) 变量微扰定义（关键：只改变量，不改参数）
# =========================
def build_variable_perturbations(base_row: pd.Series) -> List[Tuple[str, Dict[str, Any]]]:
    """
    给定某个基准情境（base_row），返回一组“变量扰动后的情境输入”。
    注意：这里做的是 controls 级别修改（brightness/cpu/net/gps/bg...），不是 pp 参数修改。
    """
    r = base_row.to_dict()

    def make(name: str, **updates) -> Tuple[str, Dict[str, Any]]:
        rr = dict(r)
        rr.update(updates)
        # 对关键变量做安全夹紧
        rr["brightness"]   = clamp01(float(rr["brightness"]))
        rr["cpu_util"]     = clamp01(float(rr["cpu_util"]))
        rr["bg_intensity"] = clamp01(float(rr["bg_intensity"]))
        rr["screen_on"]    = 1.0 if float(rr["screen_on"]) >= 0.5 else 0.0
        rr["gps_on"]       = 1.0 if float(rr["gps_on"]) >= 0.5 else 0.0
        rr["bg_on"]        = 1.0 if float(rr["bg_on"])  >= 0.5 else 0.0
        rr["net_mode"]     = str(rr["net_mode"])
        return name, rr

    pert = []

    # 0) baseline（必须保留，用于Δ计算）
    pert.append(make("baseline"))

    # 1) brightness 上调 +0.05 / 下调 -0.05
    pert.append(make("brightness_up",   brightness=float(r["brightness"]) + 0.05))
    pert.append(make("brightness_down", brightness=float(r["brightness"]) - 0.05))

    # 2) cpu_util 上调 +0.05 / 下调 -0.05
    pert.append(make("cpu_up",   cpu_util=float(r["cpu_util"]) + 0.05))
    pert.append(make("cpu_down", cpu_util=float(r["cpu_util"]) - 0.05))

    # 3) network: 切换到 4G / 5G（其余不变）
    pert.append(make("net_4g", net_mode="4g"))
    pert.append(make("net_5g", net_mode="5g"))

    # 4) gps 开启
    pert.append(make("gps_on", gps_on=1.0))

    # 5) 后台更强：bg_on=1 且强度至少到 0.20（对齐你截图里的 0.05->0.20 思路）
    pert.append(make("bg_on_more_intensity", bg_on=1.0, bg_intensity=max(0.20, float(r["bg_intensity"]))))

    return pert


# =========================
# 5) 批量评估：scenario × perturbation
# =========================
def evaluate_scenarios_with_perturbations(pp: PowerParams, scenarios: pd.DataFrame) -> pd.DataFrame:
    """
    返回长表 df：
    每行 = (scenario, perturbation)
    字段含：
      - P_total
      - ΔP_total（相对该 scenario 的 baseline）
      - 各分量功率
      - 各分量占比（share_%）
    """
    records = []

    for _, row in scenarios.iterrows():
        perts = build_variable_perturbations(row)

        # 先算 baseline
        base_name, base_inputs = perts[0]
        base_comps = power_components_step(
            pp=pp,
            area_ratio=float(base_inputs["area_ratio"]),
            screen_on=float(base_inputs["screen_on"]),
            brightness=float(base_inputs["brightness"]),
            cpu_util=float(base_inputs["cpu_util"]),
            net_mode=str(base_inputs["net_mode"]),
            gps_on=float(base_inputs["gps_on"]),
            bg_on=float(base_inputs["bg_on"]),
            bg_intensity=float(base_inputs["bg_intensity"]),
        )
        P_base = float(base_comps["P_total"])

        for pname, inp in perts:
            comps = power_components_step(
                pp=pp,
                area_ratio=float(inp["area_ratio"]),
                screen_on=float(inp["screen_on"]),
                brightness=float(inp["brightness"]),
                cpu_util=float(inp["cpu_util"]),
                net_mode=str(inp["net_mode"]),
                gps_on=float(inp["gps_on"]),
                bg_on=float(inp["bg_on"]),
                bg_intensity=float(inp["bg_intensity"]),
            )
            P = float(comps["P_total"])
            dP = P - P_base

            denom = P if P > 1e-12 else 1.0
            shares = {f"{k}_share": float(comps[k]) / denom for k in COMP_COLS}

            rec = {
                "scenario": str(row["scenario"]),
                "perturbation": pname,
                "P_total": P,
                "dP_total": dP,
            }
            # 分量功率
            for k in COMP_COLS:
                rec[k] = float(comps[k])
            # 分量占比
            rec.update(shares)

            records.append(rec)

    return pd.DataFrame(records)


# =========================
# 6) 图1：Bar - Sensitivity of Total Power (ΔP_total)
# =========================
def plot_sensitivity_bar(df: pd.DataFrame, title: str = "Sensitivity of Total Power (ΔP_total)"):
    """
    以“扰动类型”为横轴，y 为跨所有 scenarios 的平均 ΔP_total（相对每个 scenario 的 baseline）。
    颜色：Δ>0 用偏红（暖），Δ<0 用偏蓝绿（冷）。柱顶标注数值。
    """
    sub = df[df["perturbation"] != "baseline"].copy()
    g = sub.groupby("perturbation")["dP_total"].mean().sort_values(ascending=False)

    names = g.index.tolist()
    vals = g.values

    # 正负颜色区分（保持海洋+森系，不用大红大绿）
    colors = []
    for v in vals:
        if v >= 0:
            colors.append(OCEAN_FOREST["sand"])        # 暖色：沙色偏暖
        else:
            colors.append(OCEAN_FOREST["seafoam"])     # 冷色：海沫绿

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    ax.bar(x, vals, color=colors, edgecolor=OCEAN_FOREST["ink"], linewidth=0.6, alpha=0.95)

    ax.axhline(0, color=OCEAN_FOREST["ink"], linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("ΔP_total (W)")
    ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, color=OCEAN_FOREST["grid"])

    # 数值标注
    for i, v in enumerate(vals):
        ax.text(i, v + (0.02 if v >= 0 else -0.02), f"{v:+.3f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color=OCEAN_FOREST["ink"])

    plt.tight_layout()
    _save_fig("1_bar_sensitivity_dP_total.png")
    plt.show()


# =========================
# 7) 图2：Heatmap - Component Power Share
# =========================
def plot_component_share_heatmap(df: pd.DataFrame, title: str = "Component Power Share (Heatmap)"):
    """
    行：perturbation（含 baseline 也可）
    列：components
    值：跨 scenarios 的平均 share（%），每格标注百分比
    """
    # 计算平均占比
    share_cols = [f"{k}_share" for k in COMP_COLS]
    pivot = df.groupby("perturbation")[share_cols].mean()

    # 转成百分比矩阵
    M = (pivot.values * 100.0)
    row_names = pivot.index.tolist()
    col_names = [c.replace("_share", "") for c in pivot.columns.tolist()]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(M, aspect="auto")

    ax.set_yticks(np.arange(len(row_names)))
    ax.set_yticklabels(row_names)
    ax.set_xticks(np.arange(len(col_names)))
    ax.set_xticklabels(col_names, rotation=25, ha="right")

    ax.set_title(title)
    ax.set_xlabel("Component")
    ax.set_ylabel("Perturbation")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Share (%)")

    # 每格标注
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8, color=OCEAN_FOREST["ink"])

    plt.tight_layout()
    _save_fig("2_heatmap_component_share.png")
    plt.show()


# =========================
# 8) 图3：Stacked Bar - Power Component Breakdown（按场景输出）
# =========================
def plot_stacked_components_grid(
    df: pd.DataFrame,
    scenarios: List[str],
    ncols: int = 5,
    title: str = "Power Component Breakdown by Scenario (Stacked Bars)"
):
    """
    把“图3：分量堆叠柱状图”合成一张大图：
    - 每个子图对应一个 scenario
    - 子图内：baseline + 各 perturbation 的分量堆叠柱
    - 每根柱顶标注 P_total 数值
    图表中只出现英文。
    """
    # perturbation 顺序：baseline 置顶
    perts_all = df["perturbation"].unique().tolist()
    order = ["baseline"] + [p for p in perts_all if p != "baseline"]

    n = len(scenarios)
    ncols = int(max(1, ncols))
    nrows = int(np.ceil(n / ncols))

    # 画布尺寸：每个子图大约 4.6×3.2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.6 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    # 为了全图可比性：统一 y 轴上限（取所有 scenario & perturbation 的最大 P_total 略放大）
    y_max = float(df["P_total"].max())
    y_lim = y_max * 1.15 + 1e-9

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        sub = df[df["scenario"] == sc].copy()

        # 有些场景可能缺少某些扰动（理论上不该），这里做稳健对齐
        existing = sub["perturbation"].unique().tolist()
        local_order = [p for p in order if p in existing]

        sub = sub.set_index("perturbation").loc[local_order].reset_index()

        x = np.arange(len(local_order))
        bottom = np.zeros(len(local_order), dtype=float)

        # 堆叠分量
        for comp in COMP_COLS:
            ax.bar(
                x,
                sub[comp].values,
                bottom=bottom,
                color=COMP_COLOR.get(comp, OCEAN_FOREST["stone"]),
                edgecolor=OCEAN_FOREST["ink"],
                linewidth=0.35,
                alpha=0.95
            )
            bottom += sub[comp].values

        # 柱顶标注总功耗
        for i, p in enumerate(local_order):
            P = float(sub.loc[sub["perturbation"] == p, "P_total"].values[0])
            ax.text(
                i, P + 0.02 * y_lim, f"{P:.2f}W",
                ha="center", va="bottom",
                fontsize=8, color=OCEAN_FOREST["ink"], rotation=90
            )

        ax.set_title(f"Scenario: {sc}", fontsize=10)
        ax.set_ylim(0, y_lim)
        ax.grid(axis="y", alpha=0.20, color=OCEAN_FOREST["grid"])

        # x 轴标签太长：只显示少量或旋转
        ax.set_xticks(x)
        ax.set_xticklabels(local_order, rotation=45, ha="right", fontsize=8)

        # 只在第一列显示 y label，减少拥挤
        if idx % ncols == 0:
            ax.set_ylabel("Power (W)")
        else:
            ax.set_ylabel("")

    # 多余子图隐藏
    for j in range(n, len(axes)):
        axes[j].axis("off")

    # 全局标题
    fig.suptitle(title, fontsize=14, y=1.02)

    # 做一个“全图共享图例”（放在下方）
    handles = []
    labels = []
    for comp in COMP_COLS:
        h = plt.Rectangle((0, 0), 1, 1, color=COMP_COLOR.get(comp, OCEAN_FOREST["stone"]))
        handles.append(h)
        labels.append(comp)
    fig.legend(handles, labels, ncol=5, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    _save_fig("3_stacked_components_grid.png")
    plt.show()
# =========================
# 9) 图4：Ridgeline - Power Distribution under Perturbations
# =========================
def _smooth_hist_density(x: np.ndarray, bins: int = 60, sigma: float = 1.6) -> Tuple[np.ndarray, np.ndarray]:
    """
    用直方图 + 高斯核卷积近似 KDE KDE（避免依赖 scipy）
    返回：grid_x, density_y（已平滑）
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])

    lo, hi = float(np.min(x)), float(np.max(x))
    if abs(hi - lo) < 1e-9:
        lo -= 1e-3
        hi += 1e-3

    hist, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # 构造高斯核
    ksize = int(max(5, sigma * 6))
    if ksize % 2 == 0:
        ksize += 1
    t = np.linspace(-(ksize // 2), (ksize // 2), ksize)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= np.sum(kernel)

    smooth = np.convolve(hist, kernel, mode="same")
    return centers, smooth


def plot_ridgeline(df: pd.DataFrame, title: str = "Power Distribution under Perturbations (Ridgeline)"):
    """
    每条“山脊”表示某个 perturbation 下，跨 scenarios 的 P_total 分布（平滑直方图近似）
    并在每条曲线上标注中位数
    """
    perts = df["perturbation"].unique().tolist()

    # 为了可读性：baseline 放最上面
    if "baseline" in perts:
        perts = ["baseline"] + [p for p in perts if p != "baseline"]

    fig, ax = plt.subplots(figsize=(12, 7))

    # 计算全局 x 范围，统一坐标
    all_p = df["P_total"].values
    x_min, x_max = float(np.min(all_p)), float(np.max(all_p))
    pad = 0.05 * (x_max - x_min + 1e-9)
    x_min -= pad
    x_max += pad

    y_offset = 0.0
    gap = 1.0  # 每条山脊之间的间距（纵向）

    for i, p in enumerate(perts):
        vals = df.loc[df["perturbation"] == p, "P_total"].values
        gx, gy = _smooth_hist_density(vals, bins=70, sigma=1.8)

        # 统一裁剪到全局范围
        mask = (gx >= x_min) & (gx <= x_max)
        gx = gx[mask]
        gy = gy[mask]

        # 标准化高度，避免某条曲线过高
        maxy = float(np.max(gy)) if gy.size > 0 else 1.0
        gy_norm = (gy / maxy) * 0.8  # 控制高度

        # 颜色：baseline 用深海蓝，其它用森系渐变
        if p == "baseline":
            line_color = OCEAN_FOREST["ocean_deep"]
            fill_color = OCEAN_FOREST["ocean_light"]
        else:
            line_color = OCEAN_FOREST["forest_deep"]
            fill_color = OCEAN_FOREST["forest_light"]

        ax.plot(gx, gy_norm + y_offset, color=line_color, linewidth=1.2)
        ax.fill_between(gx, y_offset, gy_norm + y_offset, color=fill_color, alpha=0.45)

        # 标注中位数
        med = float(np.median(vals)) if vals.size > 0 else np.nan
        if np.isfinite(med):
            ax.text(med, y_offset + 0.85, f"median={med:.3f}W",
                    ha="center", va="bottom", fontsize=9, color=OCEAN_FOREST["ink"])

        # y 轴标签（用文字标注每条山脊）
        ax.text(x_min, y_offset + 0.35, p, ha="left", va="center",
                fontsize=9, color=OCEAN_FOREST["ink"])

        y_offset += gap

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.2, y_offset + 0.6)
    ax.set_yticks([])  # 山脊图通常不显示 y 刻度
    ax.set_xlabel("P_total (W)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25, color=OCEAN_FOREST["grid"])

    plt.tight_layout()
    _save_fig("4_ridgeline_power_distribution.png")
    plt.show()


# =========================
# 10) 输出辅助表：用于论文写作/解释
# =========================
def summarize_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    输出一个“敏感性汇总表”：
      - 每个 perturbation 的平均 ΔP_total
      - 最大影响场景（ΔP 最大）与最小影响场景（ΔP 最小）
    """
    sub = df[df["perturbation"] != "baseline"].copy()

    rows = []
    for p, g in sub.groupby("perturbation"):
        mean_dp = float(g["dP_total"].mean())
        idx_max = g["dP_total"].idxmax()
        idx_min = g["dP_total"].idxmin()

        rows.append({
            "perturbation": p,
            "mean_dP_total": mean_dp,
            "most_increased_scenario": df.loc[idx_max, "scenario"],
            "max_dP_total": float(df.loc[idx_max, "dP_total"]),
            "most_decreased_scenario": df.loc[idx_min, "scenario"],
            "min_dP_total": float(df.loc[idx_min, "dP_total"]),
        })

    out = pd.DataFrame(rows).sort_values("mean_dP_total", ascending=False)
    return out


# =========================
# 11) 主程序
# =========================
def main():
    # 11.1 读取参数（只读，不改）
    json_path = "fitted_params.json"
    try:
        pp = load_params_from_json(json_path)
        print(f"[INFO] Loaded parameters from {json_path} (parameters are NOT modified).")
    except Exception as e:
        print(f"[WARN] Cannot load {json_path}: {e}")
        print("[WARN] Using built-in demo parameters (recommended: replace with your fitted_params.json).")
        pp = PowerParams(
            p0=0.40,
            ps0=0.80,
            ps1=1.20,
            pc0=0.25,
            pc1=1.10,
            gamma=1.60,
            pwifi=0.35,
            p4g=0.55,
            p5g=0.75,
            pgps=0.40,
            pbg0=0.15,
            pbg1=0.30,
        )

    # 11.2 情境输入
    scenarios = default_scenarios()

    # 11.3 批量评估（核心：scenario × perturbation）
    df = evaluate_scenarios_with_perturbations(pp, scenarios)

    # 11.4 图1：敏感性条形图（ΔP_total）
    plot_sensitivity_bar(df, title="Sensitivity of Total Power (ΔP_total)")

    # 11.5 图2：分量占比热力图（share %）
    plot_component_share_heatmap(df, title="Component Power Share (Heatmap)")

    # 11.6 图3：每个 scenario 都输出一张堆叠分量图（保留“不同情境下变量微调的影响”）
    plot_stacked_components_grid(
        df,
        scenarios=scenarios["scenario"].tolist(),
        ncols=5,
        title="Power Component Breakdown by Scenario (Stacked Bars)"
    )

    # 11.7 图4：山脊图（不同扰动下功耗分布）
    plot_ridgeline(df, title="Power Distribution under Perturbations (Ridgeline)")

    # 11.8 输出汇总表（方便你写论文）
    summary = summarize_table(df)
    print("\n========== Sensitivity Summary Table ==========")
    print(summary.to_string(index=False))

    # 也可保存 CSV
    summary_path = os.path.join(OUT_DIR, "sensitivity_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Saved summary table to: {summary_path}")

    df_path = os.path.join(OUT_DIR, "all_records_long_format.csv")
    df.to_csv(df_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved full records to: {df_path}")


if __name__ == "__main__":
    main()
