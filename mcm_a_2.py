#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第二问（Q2）：使用第一问（Q1）拟合得到的 fitted_params.json，按“用户给定的新情景数据”进行续航模拟，
并输出两类可视化图表：
1) 不同情景下的剩余使用时间（TTE）
2) 不同情景下，不同要素（屏幕/CPU/网络/GPS/后台等）对电池消耗的占比（堆叠柱状图）

重要更新（已完成）：
- 适配“上一代码输出的 json”：
  * PowerParams：不再包含 b_on_mean / u_on_mean（因为最终公式不允许中心化）
  * TempParams：改为双侧温度惩罚 alpha_minus/beta_minus/alpha_plus/beta_plus
- Q2 的功耗单步预测公式与 Q1 严格一致（最终题设公式，不用中心化）
- SOC 更新使用双侧温度惩罚：
    ΔT^-=(T_ref - T)_+, ΔT^+=(T - T_ref)_+
    eta = 1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+

运行：
  python q2_time_to_empty.py --params fitted_params.json --out tte_results.csv

输出：
- tte_results.csv
- fig_tte_S0_*.png
- fig_share_S0_*.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1) 与“上一版Q1输出json”一致的参数结构
# =========================

@dataclass
class PowerParams:
    """
    注意：此处参数对应“最终功率公式”（不含中心化项）：
      P = p0
        + Ar*I_scr*(ps0 + ps1*b)
        + I_cpu*(pc0 + pc1*u^gamma)
        + I_wifi*pwifi + I_4g*p4g + I_5g*p5g
        + I_gps*pgps
        + I_bg*pbg0
        + pbg1*a_bg
    """
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


@dataclass
class TempParams:
    """
    双侧温度惩罚（与上一版Q1输出一致）：
      ΔT^-=(T_ref - T)_+ ; ΔT^+=(T - T_ref)_+
      dS/dt = -(P/(C0*k_age))*(1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+)
    """
    T_ref: float
    alpha_minus: float
    beta_minus: float
    alpha_plus: float
    beta_plus: float


def load_params_json(path: str) -> Tuple[PowerParams, TempParams]:
    """读取上一版Q1输出的 fitted_params.json。"""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pp = PowerParams(**payload["PowerParams"])
    tp = TempParams(**payload["TempParams"])
    return pp, tp


# =========================
# 2) 工具函数
# =========================

def _safe_filename(x: str) -> str:
    """把字符串转成安全文件名片段。"""
    s = str(x).replace(" ", "_").replace("/", "_").replace("\\", "_")
    s = s.replace(".", "p")
    return s


def pos_part(x: float) -> float:
    """正部函数 [x]_+。"""
    return float(x) if float(x) > 0.0 else 0.0


# =========================
# 3) 功耗模型：单步预测（严格题设最终公式，不做中心化）
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
    I_bg = 1.0 if float(bg_on) >= 0.5 else 0.0

    nm = (net_mode or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g = 1.0 if nm == "4g" else 0.0
    I_5g = 1.0 if nm == "5g" else 0.0

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
    """同一套功耗公式下的分量拆解（用于能量占比）。"""
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

    comps = {}
    comps["P_base"] = float(pp.p0)
    comps["P_scr"] = float(area_ratio) * I_scr * (float(pp.ps0) + float(pp.ps1) * b)
    comps["P_cpu"] = I_cpu * (float(pp.pc0) + float(pp.pc1) * ug)
    comps["P_wifi"] = I_wifi * float(pp.pwifi)
    comps["P_4g"] = I_4g * float(pp.p4g)
    comps["P_5g"] = I_5g * float(pp.p5g)
    comps["P_gps"] = I_gps * float(pp.pgps)
    comps["P_bg_sw"] = I_bg * float(pp.pbg0)
    comps["P_bg_int"] = float(pp.pbg1) * float(np.clip(bg_intensity, 0.0, 1.0))
    comps["P_total"] = sum(comps.values())
    return comps


# =========================
# 4) 使用模式（情景）定义（你给的 그대로 保留）
# =========================

def default_modes() -> Dict[str, Dict[str, Any]]:
    """
    你提供的新情景集合（controls常量）。
    备注：
    - 如果 mode 中给了 temp_c：在 thermal_enabled=0 时作为常量温度；
      在 thermal_enabled=1 时作为热模型初始温度（更合理）。
    """
    return {
        "video_wifi": {
            "controls": dict(screen_on=1, brightness=0.70, cpu_util=0.25, net_mode="wifi", gps_on=0, bg_on=0, bg_intensity=0.05),
            "temp_c": 44.9,
        },
        "shortvideo_highbright_wifi": {
            "controls": dict(screen_on=1, brightness=0.95, cpu_util=0.25, net_mode="wifi", gps_on=0, bg_on=1, bg_intensity=0.20),
            "temp_c": 45.2,
        },
        "browsing_wifi_lowbright": {
            "controls": dict(screen_on=1, brightness=0.35, cpu_util=0.22, net_mode="wifi", gps_on=0, bg_on=1, bg_intensity=0.15),
            "temp_c": 44.6,
        },
        "social_wifi": {
            "controls": dict(screen_on=1, brightness=0.50, cpu_util=0.15, net_mode="wifi", gps_on=0, bg_on=1, bg_intensity=0.30),
            "temp_c": 44.9,
        },
        "gaming_wifi": {
            "controls": dict(screen_on=1, brightness=0.80, cpu_util=0.85, net_mode="wifi", gps_on=0, bg_on=0, bg_intensity=0.00),
            "temp_c": 44.7,
        },
        "videocall_wifi": {
            "controls": dict(screen_on=1, brightness=0.70, cpu_util=0.70, net_mode="wifi", gps_on=0, bg_on=0, bg_intensity=0.00),
            "temp_c": 45.0,
        },
        "navigation_5g": {
            "controls": dict(screen_on=1, brightness=0.75, cpu_util=0.40, net_mode="5g", gps_on=1, bg_on=0, bg_intensity=0.00),
            "temp_c": 45.0,
        },
        "weaknet_4g_burst": {
            "controls": dict(screen_on=1, brightness=0.60, cpu_util=0.30, net_mode="4g", gps_on=0, bg_on=1, bg_intensity=0.85),
            "temp_c": 45.7,
        },
        "music_screenoff_wifi": {
            "controls": dict(screen_on=0, brightness=0.00, cpu_util=0.08, net_mode="wifi", gps_on=0, bg_on=1, bg_intensity=0.10),
            "temp_c": 44.7,
        },
        "idle_wifi_bgmedium": {
            "controls": dict(screen_on=0, brightness=0.00, cpu_util=0.05, net_mode="wifi", gps_on=0, bg_on=1, bg_intensity=0.45),
            "temp_c": 45.0,
        },
    }


def normalize_controls(ctrl: Dict[str, Any]) -> Dict[str, Any]:
    """补齐控制量并裁剪范围。"""
    out = dict(ctrl)
    out.setdefault("screen_on", 0)
    out.setdefault("brightness", 0.0)
    out.setdefault("cpu_util", 0.0)
    out.setdefault("net_mode", "none")
    out.setdefault("gps_on", 0)
    out.setdefault("bg_on", 0)
    out.setdefault("bg_intensity", 0.0)

    out["screen_on"] = 1.0 if float(out["screen_on"]) >= 0.5 else 0.0
    out["brightness"] = float(np.clip(float(out["brightness"]), 0.0, 1.0))
    out["cpu_util"] = float(np.clip(float(out["cpu_util"]), 0.0, 1.0))
    out["net_mode"] = str(out["net_mode"]).strip().lower()
    if out["net_mode"] not in {"wifi", "4g", "5g", "none"}:
        out["net_mode"] = "none"
    out["gps_on"] = 1.0 if float(out["gps_on"]) >= 0.5 else 0.0
    out["bg_on"] = 1.0 if float(out["bg_on"]) >= 0.5 else 0.0
    out["bg_intensity"] = float(np.clip(float(out["bg_intensity"]), 0.0, 1.0))
    return out


# =========================
# 5) 可选热模型
# =========================

def thermal_step(T_c: float, P_w: float, dt_s: float, tau_s: float, kappa_k_per_w: float, T_amb_c: float) -> float:
    """
    一阶热模型（可选）：
      dT/dt = -(T - T_amb)/tau + kappa*P
    """
    tau_s = max(float(tau_s), 1e-6)
    return float(T_c + dt_s * (-(T_c - float(T_amb_c)) / tau_s + float(kappa_k_per_w) * float(P_w)))


# =========================
# 6) 核心仿真：time-to-empty（已升级为双侧温度惩罚）
# =========================

def simulate_time_to_empty(
    pp: PowerParams,
    tp: TempParams,
    area_ratio: float,
    C0_Wh: float,
    k_age: float,
    S0: float,
    mode_def: Dict[str, Any],
    dt_s: float,
    max_sim_s: float,
    thermal_enabled: bool,
    tau_s: float,
    kappa_k_per_w: float,
    T_amb_c: float,
    T0_c: float,
) -> Dict[str, Any]:
    """
    给定初始 SOC + 情景（controls/schedule），逐步积分直到 SOC<=0，输出耗尽时间与分量占比。

    温度处理优先级（已明确）：
    - thermal_enabled=0：
        若 mode_def 提供 temp_c -> 用它作为常量温度
        否则用 T0_c 作为常量温度
    - thermal_enabled=1：
        初始温度：优先 mode_def.temp_c，否则 T0_c
        然后热模型动态更新
    """
    denom_Wh = float(C0_Wh) * float(k_age)
    if denom_Wh <= 0:
        raise ValueError("C0_Wh*k_age 必须为正")
    dt_s = float(dt_s)
    if dt_s <= 0:
        raise ValueError("dt_s 必须为正")
    dt_h = dt_s / 3600.0

    S = float(np.clip(S0, 0.0, 1.0))
    if S <= 0:
        return dict(tte_s=0.0, tte_h=0.0, steps=0, avg_power_W=0.0)

    # schedule：若未提供则按单段无限持续
    if "schedule" in mode_def and isinstance(mode_def["schedule"], list) and len(mode_def["schedule"]) > 0:
        schedule = mode_def["schedule"]
    else:
        schedule = [{"duration_s": float("inf"), "controls": mode_def.get("controls", {})}]

    # ---- 温度初始化：优先使用 mode_def.temp_c（若存在）----
    if "temp_c" in mode_def and mode_def["temp_c"] is not None:
        T_c = float(mode_def["temp_c"])
    else:
        T_c = float(T0_c)

    # 如果不启用热模型，则 T_c 恒定不变（常温常量）
    # 如果启用热模型，则 T_c 在循环中动态更新

    # 能量累计（Wh）用于占比
    E_wh = {k: 0.0 for k in ["P_base","P_scr","P_cpu","P_wifi","P_4g","P_5g","P_gps","P_bg_sw","P_bg_int"]}
    E_total_wh = 0.0

    t = 0.0
    steps = 0

    seg_idx = 0
    seg_elapsed = 0.0
    seg_dur = float(schedule[0].get("duration_s", float("inf")))
    seg_ctrl = normalize_controls(schedule[0].get("controls", {}))

    prev_S = S
    prev_t = t

    while t < float(max_sim_s) and S > 0.0:
        # 切段（循环）
        if seg_elapsed >= seg_dur:
            seg_idx = (seg_idx + 1) % len(schedule)
            seg_elapsed = 0.0
            seg_dur = float(schedule[seg_idx].get("duration_s", float("inf")))
            seg_ctrl = normalize_controls(schedule[seg_idx].get("controls", {}))

        # 1) 计算功耗（严格题设最终公式）
        P = predict_power_step(
            pp=pp,
            area_ratio=area_ratio,
            screen_on=seg_ctrl["screen_on"],
            brightness=seg_ctrl["brightness"],
            cpu_util=seg_ctrl["cpu_util"],
            net_mode=seg_ctrl["net_mode"],
            gps_on=seg_ctrl["gps_on"],
            bg_on=seg_ctrl["bg_on"],
            bg_intensity=seg_ctrl["bg_intensity"],
        )

        # 2) 温度更新（可选热模型）
        if thermal_enabled:
            T_c = thermal_step(
                T_c=T_c, P_w=P, dt_s=dt_s,
                tau_s=tau_s, kappa_k_per_w=kappa_k_per_w, T_amb_c=T_amb_c
            )
        else:
            # 不启用热模型：T_c保持常量（已在初始化时设定为 mode_def.temp_c 或 T0_c）
            pass

        # 3) 双侧温度惩罚（严格题设）
        dT_minus = pos_part(float(tp.T_ref) - float(T_c))      # ΔT^-
        dT_plus = pos_part(float(T_c) - float(tp.T_ref))       # ΔT^+

        eta = (
            1.0
            + float(tp.alpha_minus) * dT_minus
            + float(tp.beta_minus) * float(P) * dT_minus
            + float(tp.alpha_plus) * dT_plus
            + float(tp.beta_plus) * float(P) * dT_plus
        )

        # 4) 欧拉更新 SOC
        prev_S = S
        prev_t = t

        dS = (float(P) * float(eta) * dt_h) / denom_Wh
        S = float(S - dS)

        t = float(t + dt_s)
        seg_elapsed = float(seg_elapsed + dt_s)
        steps += 1

        # 5) 归因累计（Wh）：用“原始功耗分量”累计占比
        comps = power_components_step(
            pp=pp,
            area_ratio=area_ratio,
            screen_on=seg_ctrl["screen_on"],
            brightness=seg_ctrl["brightness"],
            cpu_util=seg_ctrl["cpu_util"],
            net_mode=seg_ctrl["net_mode"],
            gps_on=seg_ctrl["gps_on"],
            bg_on=seg_ctrl["bg_on"],
            bg_intensity=seg_ctrl["bg_intensity"],
        )
        for k in E_wh:
            E_wh[k] += float(comps[k]) * dt_h
        E_total_wh += float(P) * dt_h

        # 6) 跨过0：线性插值求更精确耗尽时间
        if S <= 0.0:
            S_now = S
            if prev_S > 0 and (prev_S - S_now) > 1e-12:
                frac = prev_S / (prev_S - S_now)
                t_empty = prev_t + frac * dt_s
            else:
                t_empty = t

            avg_power = (E_total_wh / (t_empty / 3600.0)) if t_empty > 1e-9 else 0.0
            ratios = {f"share_{k}": (E_wh[k] / E_total_wh if E_total_wh > 1e-12 else 0.0) for k in E_wh}

            return dict(
                tte_s=float(t_empty),
                tte_h=float(t_empty) / 3600.0,
                steps=int(steps),
                avg_power_W=float(avg_power),
                E_total_Wh=float(E_total_wh),
                T_end_c=float(T_c),
                **{f"E_{k}_Wh": float(v) for k, v in E_wh.items()},
                **ratios,
            )

    # 未在 max_sim_s 内耗尽
    avg_power = (E_total_wh / (t / 3600.0)) if t > 1e-9 else 0.0
    ratios = {f"share_{k}": (E_wh[k] / E_total_wh if E_total_wh > 1e-12 else 0.0) for k in E_wh}
    return dict(
        tte_s=float("nan"),
        tte_h=float("nan"),
        steps=int(steps),
        avg_power_W=float(avg_power),
        E_total_Wh=float(E_total_wh),
        T_end_c=float(T_c),
        **{f"E_{k}_Wh": float(v) for k, v in E_wh.items()},
        **ratios,
        note=f"在 max_sim_s={max_sim_s} 秒内未耗尽",
    )


# =========================
# 7) 可视化
# =========================

def plot_tte_by_mode(out_df: pd.DataFrame, out_prefix: str = "fig_tte") -> None:
    """图1：每个初始 SOC 一张柱状图：不同情景的剩余时间（小时）。"""
    if "tte_h" not in out_df.columns:
        return
    for S0, sub in out_df.groupby("S0", sort=True):
        sub = sub.copy()
        sub = sub[np.isfinite(sub["tte_h"].to_numpy())]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("tte_h", ascending=False)
        modes = sub["mode"].astype(str).to_numpy()
        tte_h = sub["tte_h"].to_numpy(dtype=float)

        plt.figure()
        plt.bar(modes, tte_h)
        plt.ylabel("Time-to-Empty (hours)")
        plt.xlabel("Scenario (mode)")
        plt.title(f"TTE by Scenario (S0={S0:.2f})")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_S0_{_safe_filename(f'{S0:.2f}')}.png", dpi=200)
        plt.close()


def plot_share_stacked(out_df: pd.DataFrame, out_prefix: str = "fig_share") -> None:
    """图2：每个初始 SOC 一张堆叠柱状图：不同情景下分量消耗占比（share_*）。"""
    share_cols = [c for c in out_df.columns if c.startswith("share_")]
    if not share_cols:
        return

    preferred_order = [
        "share_P_base",
        "share_P_scr",
        "share_P_cpu",
        "share_P_wifi",
        "share_P_4g",
        "share_P_5g",
        "share_P_gps",
        "share_P_bg_sw",
        "share_P_bg_int",
    ]
    share_cols_sorted = [c for c in preferred_order if c in share_cols] + [c for c in share_cols if c not in preferred_order]

    for S0, sub in out_df.groupby("S0", sort=True):
        sub = sub.copy()
        ok = np.ones(len(sub), dtype=bool)
        for c in share_cols_sorted:
            ok &= np.isfinite(sub[c].to_numpy(dtype=float))
        sub = sub.loc[ok]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("mode")
        modes = sub["mode"].astype(str).to_numpy()

        plt.figure()
        bottom = np.zeros(len(sub), dtype=float)
        for c in share_cols_sorted:
            vals = sub[c].to_numpy(dtype=float)
            plt.bar(modes, vals, bottom=bottom, label=c.replace("share_", ""))
            bottom += vals

        plt.ylabel("Energy Share (0~1)")
        plt.xlabel("Scenario (mode)")
        plt.title(f"Energy Share by Component (S0={S0:.2f})")
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1.0)
        plt.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_S0_{_safe_filename(f'{S0:.2f}')}.png", dpi=200)
        plt.close()


# =========================
# 8) 命令行与批量运行
# =========================

def parse_list_floats(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    out: List[float] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(float(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=str, default="fitted_params.json", help="第一问输出的参数 json（上一版双侧温度）")
    ap.add_argument("--out", type=str, default="tte_results.csv", help="输出结果 CSV")

    ap.add_argument("--C0_Wh", type=float, default=15.0, help="电池标称可用能量（Wh）")
    ap.add_argument("--k_age", type=float, default=1.0, help="老化系数（无量纲）")
    ap.add_argument("--area_ratio", type=float, default=1.0, help="屏幕面积比 A_screen/A_ref")

    ap.add_argument("--dt_s", type=float, default=1.0, help="仿真步长（秒）")
    ap.add_argument("--max_sim_s", type=float, default=48 * 3600.0, help="最大仿真时长（秒）")
    ap.add_argument("--soc_list", type=str, default="0.2,0.5,0.8", help="初始 SOC 列表，逗号分隔")

    # 热模型（可选）
    ap.add_argument("--thermal_enabled", type=int, default=1, help="是否启用热模型（1=启用，0=关闭）")
    ap.add_argument("--tau_s", type=float, default=1200.0, help="热时间常数 tau（秒）")
    ap.add_argument("--kappa_k_per_w", type=float, default=0.06, help="热增益 kappa（K/(W*s)）")
    ap.add_argument("--T_amb_c", type=float, default=25.0, help="环境温度（°C）")
    ap.add_argument("--T0_c", type=float, default=30.0, help="热模型默认初始温度（°C）；若 mode 给temp_c则优先用mode")

    ap.add_argument("--modes", type=str, default="", help="要运行的模式名列表，逗号分隔；空=全部默认模式")

    # 图表
    ap.add_argument("--plot", type=int, default=1, help="是否输出图表（1=输出，0=不输出）")
    ap.add_argument("--fig_prefix_tte", type=str, default="fig_tte", help="TTE 图文件名前缀")
    ap.add_argument("--fig_prefix_share", type=str, default="fig_share", help="share 图文件名前缀")

    args = ap.parse_args()

    pp, tp = load_params_json(args.params)

    modes = default_modes()
    if args.modes.strip():
        names = [x.strip() for x in args.modes.split(",") if x.strip()]
        missing = [n for n in names if n not in modes]
        if missing:
            raise ValueError(f"未知模式：{missing}。可选：{list(modes.keys())}")
        modes = {k: modes[k] for k in names}

    soc_list = parse_list_floats(args.soc_list)
    if not soc_list:
        raise ValueError("soc_list 为空，请提供例如 0.2,0.5,0.8")
    for s in soc_list:
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"soc_list 中存在非法 SOC：{s}")

    rows = []
    for mode_name, mode_def in modes.items():
        for S0 in soc_list:
            res = simulate_time_to_empty(
                pp=pp,
                tp=tp,
                area_ratio=float(args.area_ratio),
                C0_Wh=float(args.C0_Wh),
                k_age=float(args.k_age),
                S0=float(S0),
                mode_def=mode_def,
                dt_s=float(args.dt_s),
                max_sim_s=float(args.max_sim_s),
                thermal_enabled=bool(int(args.thermal_enabled)),
                tau_s=float(args.tau_s),
                kappa_k_per_w=float(args.kappa_k_per_w),
                T_amb_c=float(args.T_amb_c),
                T0_c=float(args.T0_c),
            )

            row = dict(
                mode=mode_name,
                S0=float(S0),
                params_file=args.params,
                C0_Wh=float(args.C0_Wh),
                k_age=float(args.k_age),
                area_ratio=float(args.area_ratio),
                dt_s=float(args.dt_s),
                thermal_enabled=int(bool(int(args.thermal_enabled))),
                tau_s=float(args.tau_s),
                kappa_k_per_w=float(args.kappa_k_per_w),
                T_amb_c=float(args.T_amb_c),
                T0_c=float(args.T0_c),
                # 记录温度参数（双侧）
                T_ref=float(tp.T_ref),
                alpha_minus=float(tp.alpha_minus),
                beta_minus=float(tp.beta_minus),
                alpha_plus=float(tp.alpha_plus),
                beta_plus=float(tp.beta_plus),
                # 记录功耗参数关键指数
                gamma=float(pp.gamma),
            )
            row.update(res)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"[OK] 已写出：{args.out}")
    print(out_df[["mode", "S0", "tte_s", "tte_h", "avg_power_W", "T_end_c"]].to_string(index=False))

    if int(args.plot) == 1:
        plot_tte_by_mode(out_df, out_prefix=str(args.fig_prefix_tte))
        plot_share_stacked(out_df, out_prefix=str(args.fig_prefix_share))
        print("[OK] 已输出图表：")
        print(f"  - {args.fig_prefix_tte}_S0_*.png（不同情景下剩余时间）")
        print(f"  - {args.fig_prefix_share}_S0_*.png（不同情景下分量消耗占比）")


if __name__ == "__main__":
    main()
