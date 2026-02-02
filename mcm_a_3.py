# -*- coding: utf-8 -*-
"""
负向微扰敏感性分析（基于 fitted_params.json 的模型参数 pp + tp）
- 固定控制量：video_wifi
- 对 pp / tp 中各参数分别做轻微负向修改（默认 -5%）
- 比较 P_total、eta(T,P) 与 P_eff = P_total*eta，以及各分量占比变化
- 输出英文标题的表格与多种图表（含 ridgeline），注释为中文
- 导出图片到 figs/ 目录
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 0) 你的模型参数结构
# =========================
@dataclass
class PowerParams:
    p0: float
    ps0: float
    ps1: float
    pc0: float
    pc1: float
    pwifi: float
    p4g: float
    p5g: float
    pgps: float
    pbg0: float
    pbg1: float
    gamma: float
    b_on_mean: float
    u_on_mean: float

@dataclass
class TempParams:
    T_ref: float
    alpha_minus: float
    beta_minus: float
    alpha_plus: float
    beta_plus: float


# =========================
# 1) 你给的模型（保持原样）
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
    b = float(np.clip(brightness, 0.0, 1.0))
    u = float(np.clip(cpu_util, 0.0, 1.0))

    I_scr = 1.0 if float(screen_on) >= 0.5 else 0.0
    I_cpu = 1.0 if u > 1e-3 else 0.0
    I_gps = 1.0 if float(gps_on) >= 0.5 else 0.0
    I_bg = 1.0 if float(bg_on) >= 0.5 else 0.0

    nm = (net_mode or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g = 1.0 if nm == "4g" else 0.0
    I_5g = 1.0 if nm == "5g" else 0.0

    ug = u ** float(pp.gamma)
    b_center = b - float(pp.b_on_mean)
    u_center = ug - float(pp.u_on_mean)

    P = float(pp.p0)
    P += float(area_ratio) * I_scr * (float(pp.ps0) + float(pp.ps1) * b_center)
    P += I_cpu * (float(pp.pc0) + float(pp.pc1) * u_center)
    P += I_wifi * float(pp.pwifi) + I_4g * float(pp.p4g) + I_5g * float(pp.p5g)
    P += I_gps * float(pp.pgps)
    # 与 Q1 一致：强度项不额外乘 I_bg
    P += I_bg * float(pp.pbg0) + float(pp.pbg1) * float(np.clip(bg_intensity, 0.0, 1.0))

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
    b = float(np.clip(brightness, 0.0, 1.0))
    u = float(np.clip(cpu_util, 0.0, 1.0))

    I_scr = 1.0 if float(screen_on) >= 0.5 else 0.0
    I_cpu = 1.0 if u > 1e-3 else 0.0
    I_gps = 1.0 if float(gps_on) >= 0.5 else 0.0
    I_bg = 1.0 if float(bg_on) >= 0.5 else 0.0

    nm = (net_mode or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g = 1.0 if nm == "4g" else 0.0
    I_5g = 1.0 if nm == "5g" else 0.0

    ug = u ** float(pp.gamma)
    b_center = b - float(pp.b_on_mean)
    u_center = ug - float(pp.u_on_mean)

    comps = {}
    comps["P_base"] = float(pp.p0)
    comps["P_scr"] = float(area_ratio) * I_scr * (float(pp.ps0) + float(pp.ps1) * b_center)
    comps["P_cpu"] = I_cpu * (float(pp.pc0) + float(pp.pc1) * u_center)
    comps["P_wifi"] = I_wifi * float(pp.pwifi)
    comps["P_4g"] = I_4g * float(pp.p4g)
    comps["P_5g"] = I_5g * float(pp.p5g)
    comps["P_gps"] = I_gps * float(pp.pgps)
    comps["P_bg_sw"] = I_bg * float(pp.pbg0)
    comps["P_bg_int"] = float(pp.pbg1) * float(np.clip(bg_intensity, 0.0, 1.0))
    comps["P_total"] = sum(comps.values())
    return comps


# =========================
# 2) 读取 fitted_params.json（新的结构：PowerParams + TempParams）
# =========================
def load_params(json_path: str) -> Tuple[PowerParams, TempParams]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("fitted_params.json 顶层应为 dict")

    if "PowerParams" not in data or "TempParams" not in data:
        raise ValueError("新的 json 格式要求同时包含 PowerParams 与 TempParams")

    p = dict(data["PowerParams"])
    t = dict(data["TempParams"])

    # 兼容：新 json 中缺 b_on_mean / u_on_mean，则默认 0.0
    missing_means = []
    if "b_on_mean" not in p:
        p["b_on_mean"] = 0.0
        missing_means.append("b_on_mean")
    if "u_on_mean" not in p:
        p["u_on_mean"] = 0.0
        missing_means.append("u_on_mean")
    if missing_means:
        print(f"[WARN] PowerParams 缺少 {missing_means}，已默认设为 0.0（为兼容 predict_power_step 的中心化项）。")

    # 必要字段检查（功耗）
    need_p = {"p0","ps0","ps1","pc0","pc1","gamma","pwifi","p4g","p5g","pgps","pbg0","pbg1","b_on_mean","u_on_mean"}
    miss_p = sorted(list(need_p - set(p.keys())))
    if miss_p:
        raise ValueError(f"PowerParams 缺少字段：{miss_p}")

    # 必要字段检查（温度）
    need_t = {"T_ref","alpha_minus","beta_minus","alpha_plus","beta_plus"}
    miss_t = sorted(list(need_t - set(t.keys())))
    if miss_t:
        raise ValueError(f"TempParams 缺少字段：{miss_t}")

    pp = PowerParams(**{k: float(p[k]) for k in need_p})
    tp = TempParams(**{k: float(t[k]) for k in need_t})
    return pp, tp


# =========================
# 3) 固定控制量（video_wifi）
# =========================
BASE_CONTROLS = dict(
    screen_on=1, brightness=0.70,
    cpu_util=0.25, net_mode="wifi",
    gps_on=0, bg_on=0, bg_intensity=0.05
)
BASE_TEMP_C = 44.9  # 用于 eta(T,P)
AREA_RATIO_DEFAULT = 1.0


# =========================
# 4) 温度惩罚：eta(T,P)
# =========================
def pos_part(x: float) -> float:
    return float(x) if float(x) > 0.0 else 0.0

def compute_eta(tp: TempParams, T_c: float, P_w: float) -> float:
    """
    eta(T,P) = 1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+
    """
    dT_minus = pos_part(float(tp.T_ref) - float(T_c))
    dT_plus  = pos_part(float(T_c) - float(tp.T_ref))
    eta = (
        1.0
        + float(tp.alpha_minus) * dT_minus
        + float(tp.beta_minus)  * float(P_w) * dT_minus
        + float(tp.alpha_plus)  * dT_plus
        + float(tp.beta_plus)   * float(P_w) * dT_plus
    )
    return max(float(eta), 0.0)


# =========================
# 5) 负向微扰：对 pp/tp 的每个参数分别做轻微下调
# =========================
POWER_KEYS = [
    "p0","ps0","ps1","pc0","pc1","pwifi","p4g","p5g","pgps","pbg0","pbg1",
    "gamma","b_on_mean","u_on_mean"
]
TEMP_KEYS = ["T_ref","alpha_minus","beta_minus","alpha_plus","beta_plus"]

def perturb_negative(obj: Any, key: str, frac: float = 0.05) -> Any:
    """
    对 dataclass 的指定字段做负向相对扰动：new = old*(1-frac)
    对 gamma 做下调时保证 > 0；对温度相关系数也保证非负（避免奇异）。
    """
    d = obj.__dict__.copy()
    old = float(d[key])
    new = old * (1.0 - frac)

    if key == "gamma":
        new = max(new, 1e-3)

    # 温度系数/参考温度做简单安全约束
    if key in {"alpha_minus","beta_minus","alpha_plus","beta_plus"}:
        new = max(new, 0.0)

    # T_ref 允许下调（物理上是参考温度），但做个合理截断避免极端
    if key == "T_ref":
        new = np.clip(new, -50.0, 80.0)

    d[key] = float(new)
    return obj.__class__(**d)


# =========================
# 6) 计算 baseline 与各参数扰动后的结果
# =========================
COMP_KEYS = ["P_base","P_scr","P_cpu","P_wifi","P_4g","P_5g","P_gps","P_bg_sw","P_bg_int"]

def eval_one(pp: PowerParams, tp: TempParams, controls: Dict[str, Any], T_c: float) -> Dict[str, float]:
    """计算一次分量并补充占比 + 温度 eta + P_eff"""
    comps = power_components_step(
        pp=pp,
        area_ratio=AREA_RATIO_DEFAULT,
        screen_on=controls["screen_on"],
        brightness=controls["brightness"],
        cpu_util=controls["cpu_util"],
        net_mode=controls["net_mode"],
        gps_on=controls["gps_on"],
        bg_on=controls["bg_on"],
        bg_intensity=controls["bg_intensity"],
    )
    P_total = float(comps["P_total"])
    eta = compute_eta(tp, T_c=float(T_c), P_w=float(P_total))
    P_eff = P_total * float(eta)

    out = dict(comps)
    out["eta"] = float(eta)
    out["P_eff"] = float(P_eff)

    for k in COMP_KEYS:
        out[k + "_share"] = (float(comps[k]) / P_total) if P_total > 0 else 0.0
    return out

def build_sensitivity_table(pp_base: PowerParams, tp_base: TempParams, frac: float = 0.05) -> pd.DataFrame:
    """
    生成敏感性分析表：
    - baseline
    - 每个 power 参数单独负向扰动（tp 不变）
    - 每个 temp 参数单独负向扰动（pp 不变）
    """
    rows = []

    # baseline
    base_res = eval_one(pp_base, tp_base, BASE_CONTROLS, BASE_TEMP_C)
    rows.append({
        "case": "baseline",
        "perturbed_group": "none",
        "perturbed_param": "none",
        "perturb_frac": 0.0,
        **base_res
    })

    # power params perturbed
    for key in POWER_KEYS:
        pp_new = perturb_negative(pp_base, key=key, frac=frac)
        res = eval_one(pp_new, tp_base, BASE_CONTROLS, BASE_TEMP_C)
        rows.append({
            "case": f"neg_pp_{key}",
            "perturbed_group": "PowerParams",
            "perturbed_param": key,
            "perturb_frac": -frac,
            **res
        })

    # temp params perturbed
    for key in TEMP_KEYS:
        tp_new = perturb_negative(tp_base, key=key, frac=frac)
        res = eval_one(pp_base, tp_new, BASE_CONTROLS, BASE_TEMP_C)
        rows.append({
            "case": f"neg_tp_{key}",
            "perturbed_group": "TempParams",
            "perturbed_param": key,
            "perturb_frac": -frac,
            **res
        })

    df = pd.DataFrame(rows)

    # 相对 baseline 的变化（P_total / eta / P_eff）
    base_total = float(df.loc[df["case"] == "baseline", "P_total"].iloc[0])
    base_eta   = float(df.loc[df["case"] == "baseline", "eta"].iloc[0])
    base_eff   = float(df.loc[df["case"] == "baseline", "P_eff"].iloc[0])

    df["delta_P_total_W"] = df["P_total"] - base_total
    df["delta_P_total_pct"] = (df["P_total"] / base_total - 1.0) * 100.0 if base_total > 0 else 0.0

    df["delta_eta"] = df["eta"] - base_eta
    df["delta_eta_pct"] = (df["eta"] / base_eta - 1.0) * 100.0 if base_eta > 0 else 0.0

    df["delta_P_eff_W"] = df["P_eff"] - base_eff
    df["delta_P_eff_pct"] = (df["P_eff"] / base_eff - 1.0) * 100.0 if base_eff > 0 else 0.0

    # 屏幕与 CPU 占比变化（百分点）
    df["delta_P_scr_share_pp"] = (df["P_scr_share"] - float(df.loc[df["case"] == "baseline", "P_scr_share"].iloc[0])) * 100.0
    df["delta_P_cpu_share_pp"] = (df["P_cpu_share"] - float(df.loc[df["case"] == "baseline", "P_cpu_share"].iloc[0])) * 100.0

    return df


# =========================
# 7) 绘图（森系渐变：高对比、导出图片、英文标题；注释中文）
# =========================
OUT_DIR = "figs"
SAVE_DPI = 300
os.makedirs(OUT_DIR, exist_ok=True)

FOREST_GRAD = [
    "#E7F3EA", "#CFE8D6", "#A8D5BA", "#6FB28E", "#2F7D5B",
    "#0E5A3F", "#1E6A6B", "#0A4A4A", "#7FAF5A",
]

MP_COLORS = {
    "P_base":  "#A8DFE0",
    "P_scr":   "#AED4D5",
    "P_cpu":   "#86CBCD",
    "P_wifi":  "#F9E2AE",
    "P_4g":    "#FBC78D",
    "P_5g":    "#FEB396",
    "P_gps":   "#F8CC88",
    "P_bg_sw": "#A6D676",
    "P_bg_int":"#FD465D",
}

def _save_fig(filename: str):
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    return path

def plot_total_delta_bar(df: pd.DataFrame, use_effective: bool = True):
    """
    图 1：每个参数负向扰动导致的 ΔP_eff（W）（默认）或 ΔP_total（W）
    """
    metric = "delta_P_eff_W" if use_effective else "delta_P_total_W"
    title = "Sensitivity (Negative Perturbation): ΔP_eff by Parameter" if use_effective else \
            "Sensitivity (Negative Perturbation): ΔP_total by Parameter"
    fname = "fig_sensitivity_delta_p_eff.png" if use_effective else "fig_sensitivity_delta_p_total.png"

    d = df[df["case"] != "baseline"].copy()
    d = d.sort_values(metric)
    labels = (d["perturbed_group"] + ":" + d["perturbed_param"]).tolist()
    vals = d[metric].tolist()

    colors = [FOREST_GRAD[i % len(FOREST_GRAD)] for i in range(len(vals))]

    plt.figure(figsize=(13, 5))
    ax = plt.gca()
    ax.axhline(0, color="#8AA18E", linewidth=1.0)
    bars = ax.bar(labels, vals, color=colors, edgecolor="#6F8A79", linewidth=0.7)

    ax.set_title(title)
    ax.set_ylabel(f"{metric} (W)")
    ax.set_xlabel("Perturbed parameter (group:param)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    y0, y1 = ax.get_ylim()
    pad = (y1 - y0) * 0.18 if y1 > y0 else 0.5
    ax.set_ylim(y0 - pad, y1 + pad)

    yr = (ax.get_ylim()[1] - ax.get_ylim()[0])
    for b, v in zip(bars, vals):
        x = b.get_x() + b.get_width()/2.0
        y = b.get_height()
        va = "bottom" if v >= 0 else "top"
        offset = 0.012 * yr
        ax.text(x, y + (offset if v >= 0 else -offset), f"{v:+.4f} W", ha="center", va=va, fontsize=9, color="#2A2A2A")

    plt.tight_layout()
    _save_fig(fname)
    plt.show()

def plot_component_share_heatmap(df: pd.DataFrame):
    """
    图 2：分量占比热力图（每个参数扰动对应各分量 share）
    """
    d = df[df["case"] != "baseline"].copy()
    d = d.sort_values("delta_P_eff_W")  # 更贴近续航

    share_cols = [k + "_share" for k in COMP_KEYS]
    M = d[share_cols].to_numpy()

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    im = ax.imshow(M, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Share of P_total")

    ax.set_title("Component Share Heatmap under Negative Parameter Perturbations")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels((d["perturbed_group"] + ":" + d["perturbed_param"]).tolist())

    ax.set_xticks(np.arange(len(share_cols)))
    ax.set_xticklabels([c.replace("_share", "") for c in share_cols], rotation=45, ha="right")

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = float(M[i, j])
            ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center", fontsize=8,
                    color="#1B1B1B" if val < 0.55 else "#FFFFFF")

    plt.tight_layout()
    _save_fig("fig_component_share_heatmap.png")
    plt.show()

def plot_stacked_components(df: pd.DataFrame):
    """
    图 3：堆叠柱状图（显示“结构”变化，基于 P_total 分量）
    """
    d = df[df["case"] != "baseline"].copy()
    d = d.sort_values("delta_P_eff_W")

    labels = (d["perturbed_group"] + ":" + d["perturbed_param"]).tolist()
    x = np.arange(len(labels))

    plt.figure(figsize=(14, 6))
    ax = plt.gca()

    bottom = np.zeros(len(labels), dtype=float)
    for k in COMP_KEYS:
        vals = d[k].values
        ax.bar(x, vals, bottom=bottom, label=k,
               color=MP_COLORS.get(k, "#CCCCCC"),
               edgecolor="#6F8A79", linewidth=0.45)
        bottom += vals

    ax.set_title("Power Component Breakdown (Stacked) for Negative Parameter Perturbations")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Perturbed parameter (group:param)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")

    totals = d["P_total"].values
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 * 1.12 if y1 > 0 else y1 + 1.0)
    yr = max(ax.get_ylim()[1] - ax.get_ylim()[0], 1e-9)
    for xi, t in zip(x, totals):
        ax.text(xi, t + 0.012 * yr, f"{t:.4f} W", ha="center", va="bottom", fontsize=9, color="#2A2A2A")

    ax.legend(ncol=3, fontsize=9, frameon=True)
    plt.tight_layout()
    _save_fig("fig_component_stacked.png")
    plt.show()

# -------- ridgeline（脊线图）--------
def _kde_1d(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    手写简易 KDE（高斯核），避免依赖额外库。
    - 使用 Silverman 带宽
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    n = max(len(x), 1)

    std = float(np.std(x)) if n > 1 else 1e-6
    bw = 1.06 * std * (n ** (-1/5)) if std > 0 else 1e-6
    bw = max(bw, 1e-6)

    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * bw * np.sqrt(2*np.pi))
    return dens

def simulate_power_distribution(pp: PowerParams, n: int = 400, seed: int = 7) -> np.ndarray:
    """
    为 ridgeline 生成“功耗分布”：围绕 baseline controls 做小幅随机扰动（模拟自然波动）
    """
    rng = np.random.default_rng(seed)

    b0 = float(BASE_CONTROLS["brightness"])
    u0 = float(BASE_CONTROLS["cpu_util"])
    bi0 = float(BASE_CONTROLS["bg_intensity"])

    brightness = np.clip(rng.normal(loc=b0, scale=0.03, size=n), 0.0, 1.0)
    cpu_util   = np.clip(rng.normal(loc=u0, scale=0.04, size=n), 0.0, 1.0)
    bg_int     = np.clip(rng.normal(loc=bi0, scale=0.03, size=n), 0.0, 1.0)

    screen_on = float(BASE_CONTROLS["screen_on"])
    net_mode  = str(BASE_CONTROLS["net_mode"])
    gps_on    = float(BASE_CONTROLS["gps_on"])
    bg_on     = float(BASE_CONTROLS["bg_on"])

    P = []
    for b, u, bii in zip(brightness, cpu_util, bg_int):
        P.append(
            predict_power_step(
                pp=pp,
                area_ratio=AREA_RATIO_DEFAULT,
                screen_on=screen_on,
                brightness=float(b),
                cpu_util=float(u),
                net_mode=net_mode,
                gps_on=gps_on,
                bg_on=bg_on,
                bg_intensity=float(bii),
            )
        )
    return np.asarray(P, dtype=float)

def plot_ridgeline_distributions(pp_base: PowerParams, tp_base: TempParams, frac: float = 0.05, top_k: int = 9, use_effective: bool = True):
    """
    图 4：Ridgeline（脊线图）
    - 选择对 P_eff（默认）或 P_total 影响最大的 top_k 个“单参数扰动”
    - 对每个扰动后的 pp 生成功耗分布（通过控制量小扰动模拟）
    - 如果 use_effective=True，则把每个样本用 eta(T,P) 转成 P_eff 分布（温度参数扰动同理）
    """
    df = build_sensitivity_table(pp_base, tp_base, frac=frac)
    d = df[df["case"] != "baseline"].copy()
    metric = "delta_P_eff_W" if use_effective else "delta_P_total_W"
    d["abs_delta"] = np.abs(d[metric])
    d = d.sort_values("abs_delta", ascending=False).head(top_k)

    # baseline 分布
    base_p = simulate_power_distribution(pp_base, n=500, seed=11)
    if use_effective:
        base_eta = np.array([compute_eta(tp_base, BASE_TEMP_C, p) for p in base_p], dtype=float)
        base_dist = base_p * base_eta
        xlab = "P_eff (W)"
        title = "Ridgeline: P_eff Distributions (Baseline vs Top Negative Parameter Perturbations)"
        fname = "fig_ridgeline_p_eff.png"
    else:
        base_dist = base_p
        xlab = "P_total (W)"
        title = "Ridgeline: P_total Distributions (Baseline vs Top Negative Parameter Perturbations)"
        fname = "fig_ridgeline_p_total.png"

    groups: List[Tuple[str, np.ndarray]] = [("baseline", base_dist)]

    for _, r in d.iterrows():
        g = str(r["perturbed_group"])
        key = str(r["perturbed_param"])

        if g == "PowerParams":
            pp_new = perturb_negative(pp_base, key=key, frac=frac)
            tp_new = tp_base
        else:
            pp_new = pp_base
            tp_new = perturb_negative(tp_base, key=key, frac=frac)

        P = simulate_power_distribution(pp_new, n=500, seed=11)
        if use_effective:
            eta = np.array([compute_eta(tp_new, BASE_TEMP_C, p) for p in P], dtype=float)
            dist = P * eta
        else:
            dist = P

        groups.append((f"neg_{g}_{key}", dist))

    all_vals = np.concatenate([g[1] for g in groups])
    xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.08 * (xmax - xmin) if xmax > xmin else 1.0
    grid = np.linspace(xmin - pad, xmax + pad, 250)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    offset = 1.3
    ridge_colors = [
        FOREST_GRAD[-2],
        FOREST_GRAD[4],
        FOREST_GRAD[3],
        FOREST_GRAD[2],
        FOREST_GRAD[1],
        FOREST_GRAD[0],
        FOREST_GRAD[6],
        FOREST_GRAD[7],
        FOREST_GRAD[8],
        FOREST_GRAD[5],
    ]

    for i, (name, dist) in enumerate(groups):
        dens = _kde_1d(dist, grid)
        dens = dens / (dens.max() + 1e-12)
        y0 = i * offset
        color = ridge_colors[i % len(ridge_colors)]

        ax.fill_between(grid, y0, y0 + dens, color=color, alpha=0.75, edgecolor="#5C7A67", linewidth=0.6)
        med = float(np.median(dist))
        ax.plot([med, med], [y0, y0 + 0.45], color="#5C7A67", linewidth=1.0)
        ax.text(med, y0 + 0.48, f"{med:.3f} W", ha="center", va="bottom", fontsize=8, color="#2A2A2A")
        ax.text(grid[0], y0 + 0.35, name, ha="left", va="center", fontsize=9, color="#2A2A2A")

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_yticks([])

    plt.tight_layout()
    _save_fig(fname)
    plt.show()


# =========================
# 8) 输出英文表格（含具体数值）并给解释提示
# =========================
def print_tables(df: pd.DataFrame):
    """
    输出英文表格：
    - Summary table: P_total / eta / P_eff and deltas
    - Component detail: all components (W) and shares (%)
    """
    d = df[df["case"] != "baseline"].copy()
    d = d.sort_values("delta_P_eff_W")

    summary_cols = [
        "perturbed_group", "perturbed_param", "perturb_frac",
        "P_total", "delta_P_total_W", "delta_P_total_pct",
        "eta", "delta_eta", "delta_eta_pct",
        "P_eff", "delta_P_eff_W", "delta_P_eff_pct",
        "P_scr_share", "delta_P_scr_share_pp",
        "P_cpu_share", "delta_P_cpu_share_pp",
    ]
    summary = d[summary_cols].copy()
    summary["P_scr_share"] = summary["P_scr_share"] * 100.0
    summary["P_cpu_share"] = summary["P_cpu_share"] * 100.0

    print("\n=== Sensitivity Summary (Negative Perturbations; PowerParams + TempParams) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    comp_cols = ["perturbed_group", "perturbed_param", "P_total"] + COMP_KEYS + [k + "_share" for k in COMP_KEYS]
    comp = d[comp_cols].copy()
    for k in COMP_KEYS:
        comp[k + "_share"] = comp[k + "_share"] * 100.0

    print("\n=== Component Breakdown (W and % of P_total) ===")
    print(comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

def explain_rules():
    print("\n=== Interpretation Guide (How to Explain the Results) ===")
    print(
        "PowerParams:\n"
        "1) Lowering p0 reduces P_base directly; shares may shift because P_total changes.\n"
        "2) Lowering ps0/ps1 mainly reduces P_scr (screen component).\n"
        "3) Lowering pc0/pc1 mainly reduces P_cpu.\n"
        "4) Lowering pwifi/p4g/p5g affects only the active network mode.\n"
        "5) Lowering pgps matters only when gps_on=1 (here gps_on=0 -> near-zero).\n"
        "6) Lowering pbg0 matters only when bg_on=1 (here bg_on=0 -> near-zero).\n"
        "7) Lowering pbg1 affects P_bg_int even if bg_on=0 (your model does not multiply intensity by I_bg).\n"
        "8) Lowering gamma changes u^gamma nonlinearity, shifting CPU mapping.\n"
        "9) Lowering b_on_mean/u_on_mean shifts centering; effect depends on current b,u relative to means.\n\n"
        "TempParams (via eta(T,P) and P_eff = P_total*eta):\n"
        "10) Lowering alpha_minus/beta_minus reduces penalty when T < T_ref.\n"
        "11) Lowering alpha_plus/beta_plus reduces penalty when T > T_ref.\n"
        "12) Lowering T_ref changes which side (hot/cold) is penalized at the given scenario temperature.\n"
    )


# =========================
# 9) 主程序
# =========================
def main():
    json_path = "fitted_params.json"  # 如需改路径，改这里
    frac = 0.05  # 负向微扰幅度（5%）

    pp_base, tp_base = load_params(json_path)

    df = build_sensitivity_table(pp_base, tp_base, frac=frac)

    print_tables(df)

    # 图表：优先按“续航相关”的 P_eff 做敏感性排序/展示
    plot_total_delta_bar(df, use_effective=True)
    plot_component_share_heatmap(df)
    plot_stacked_components(df)
    plot_ridgeline_distributions(pp_base, tp_base, frac=frac, top_k=9, use_effective=True)

    explain_rules()
    print(f"\nAll figures exported to: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
