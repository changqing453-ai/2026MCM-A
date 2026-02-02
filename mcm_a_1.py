# -*- coding: utf-8 -*-
"""
电池功耗分解 + 双侧温度惩罚SOC动力学（适配新数据 battery_dataset.csv）

硬性约束（已满足）：
1) 功率总和公式 P(t) 不允许被修改：最终计算严格使用题设给出的 P(t) 形式；
2) SOC微分方程必须使用双侧温度项（ΔT^- / ΔT^+）且形式不改；
3) 数据源改为 /mnt/data/battery_dataset.csv，并明确标注预处理步骤。
"""

import json
import typing
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import lsq_linear  # 约束最小二乘（更正规）
    HAS_SCIPY = True
except Exception:
    lsq_linear = None
    HAS_SCIPY = False


# =========================
# 1) 参数容器（模型参数结构）
# =========================

@dataclass
class PowerParams:
    """
    功耗分解模型参数（注意：这里存储的是“最终公式”的参数，不包含任何中心化均值）
    对应题设：
      P(t)=p0
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
    双侧温度惩罚模型参数（严格对应题设形式）
    ΔT^-=(T_ref - T(t))_+ ; ΔT^+=(T(t) - T_ref)_+
    dS/dt = -(P/(C0*k_age))*(1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+)
    """
    T_ref: float
    alpha_minus: float
    beta_minus: float
    alpha_plus: float
    beta_plus: float


# =========================
# 2) 基础数学工具函数
# =========================

def clip01(x: np.ndarray) -> np.ndarray:
    """裁剪到[0,1]，避免异常值导致不稳定。"""
    return np.clip(x, 0.0, 1.0)


def pos_part(x: np.ndarray) -> np.ndarray:
    """正部函数 [x]_+ = max(x,0)。"""
    return np.maximum(x, 0.0)


# =========================
# 3) 约束最小二乘：min||Ax-b||, s.t. x>=lb
# =========================

def bounded_fit(A: np.ndarray, b: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """
    解约束最小二乘：
      min_x ||A x - b||_2
      s.t.  x >= lb

    目的：
    - 保证参数非负（或不小于lb）
    - 通过 lb=eps 避免“严格为0”的数值病态
    """
    if HAS_SCIPY:
        res = lsq_linear(A, b, bounds=(lb, np.inf), method="trf")
        return res.x
    x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.maximum(x_ls, lb)


# =========================
# 4) 新数据读取与特征构造（适配 battery_dataset.csv）
# =========================

def compute_opt_temperature(temp_c: pd.Series) -> np.ndarray:
    """
    计算题设中的 T(t)（“最佳温度/有效温度”）——按你的要求“需要通过算法计算得出”。

    这里给出一个可复现、稳健的“有效温度”算法（预处理步骤会在此明确标注）：
    预处理步骤（已标注）：
    1) 强制转数值，非法转NaN；
    2) 线性插值补齐缺失（再用中位数兜底）；
    3) Winsorize 去极值：按 1%~99% 分位裁剪（防止传感器尖峰）；
    4) 平滑：rolling median(5) + EWMA(span=10) 降噪，得到 T_opt(t)。

    注：你也可以替换为更复杂的算法（Kalman/状态空间），但此处足够“算法化且可复现”。
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

    列映射（新CSV -> 模型列）：
    - timestamp -> time_s（解析datetime，转相对秒）
    - SOC -> soc（0~1，clip）
    - temperature -> temp_c（°C）
    - temperature -> temp_opt_c（算法平滑后的 T(t)）
    - brightness -> brightness（0~1，clip）
    - cpu_load -> cpu_util（0~1，clip）
    - network_type -> net_mode（wifi/4g/5g）
    - gps_enabled -> gps_on（bool->0/1）
    - num_background_apps -> bg_intensity（归一化到0~1） & bg_on（结合screen_on）
    - power_consumption_mw -> P_obs_W（mW->W）

    预处理操作（已标注）：
    1) timestamp 解析为 datetime，并按 device_id+timestamp 排序；
    2) 若存在多设备：默认选择样本数最多的 device_id（保持“单序列拟合/仿真”与原功能一致）；
    3) 数值列强制转float，缺失用插值/中位数等方式处理；
    4) brightness/cpu_load/SOC 做 clip01；
    5) screen_on 由 brightness 阈值推断（因为新CSV没有显式屏幕开关列）。
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

    # ---- 估计 dt（用于滚动斜率更稳定/输出可读）----
    if len(time_s) > 2:
        dt_med = float(np.median(np.diff(time_s)))
    else:
        dt_med = 60.0  # 兜底：假设1分钟采样

    # ---- 构造模型特征表 ----
    model = pd.DataFrame()
    model["time_s"] = time_s

    # (预处理3)(4) SOC：强制数值+裁剪
    soc = pd.to_numeric(raw["SOC"], errors="coerce").interpolate(limit_direction="both")
    soc = soc.fillna(soc.median())
    model["soc"] = clip01(soc.to_numpy(dtype=float))

    # 温度：原始 temp_c 与 算法得到的 temp_opt_c (=题设 T(t))
    temp_c = pd.to_numeric(raw["temperature"], errors="coerce").interpolate(limit_direction="both")
    temp_c = temp_c.fillna(temp_c.median())
    model["temp_c"] = temp_c.to_numpy(dtype=float)
    model["temp_opt_c"] = compute_opt_temperature(temp_c)

    # (预处理3)(4) 亮度：强制数值+裁剪
    b = pd.to_numeric(raw["brightness"], errors="coerce").interpolate(limit_direction="both")
    b = b.fillna(b.median())
    model["brightness"] = clip01(b.to_numpy(dtype=float))

    # (预处理5) screen_on：新CSV没有显式屏幕开关，只能用亮度阈值推断
    # 说明：brightness 的最小值为0.05左右，直接>0会导致全为1，因此设经验阈值 0.12
    model["screen_on"] = (model["brightness"].to_numpy() >= 0.12).astype(float)

    # CPU负载：强制数值+裁剪
    u = pd.to_numeric(raw["cpu_load"], errors="coerce").interpolate(limit_direction="both")
    u = u.fillna(u.median())
    model["cpu_util"] = clip01(u.to_numpy(dtype=float))

    # 网络模式：直接使用 network_type（统一小写）
    nm = raw["network_type"].astype(str).str.strip().str.lower()
    nm = nm.where(nm.isin(["wifi", "4g", "5g"]), other="wifi")  # 兜底
    model["net_mode"] = nm.to_numpy()

    # GPS：bool/字符串 -> 0/1
    gps = raw["gps_enabled"]
    if gps.dtype == object:
        gps = gps.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"])
    model["gps_on"] = gps.astype(float).to_numpy()

    # 后台：用后台app数量作为强度；bg_on 用“屏幕灭 + 后台app>0”推断
    nba = pd.to_numeric(raw["num_background_apps"], errors="coerce").interpolate(limit_direction="both")
    nba = nba.fillna(nba.median()).clip(lower=0.0)
    nba_np = nba.to_numpy(dtype=float)
    max_apps = float(np.max(nba_np)) if np.max(nba_np) > 0 else 1.0
    model["bg_intensity"] = clip01(nba_np / max_apps)
    model["bg_on"] = (((model["screen_on"].to_numpy() < 0.5) & (nba_np > 0.0))).astype(float)

    # 观测功耗：power_consumption_mw（mW->W）
    p_mw = pd.to_numeric(raw["power_consumption_mw"], errors="coerce").interpolate(limit_direction="both")
    p_mw = p_mw.fillna(p_mw.median()).clip(lower=0.0)
    P_obs_W = p_mw.to_numpy(dtype=float) * 1e-3

    return model, P_obs_W


# =========================
# 5) 拟合功耗参数：P_obs ≈ P_model(theta_P)
#    （最终返回参数严格对应题设“未中心化”的功率公式）
# =========================

def fit_power_params_from_Pobs(
    df: pd.DataFrame,
    P_obs_W: np.ndarray,
    gamma_grid: np.ndarray,
    area_ratio: float,
    eps: float = 1e-6
) -> PowerParams:
    """
    在 gamma_grid 上网格搜索 gamma：
      对每个 gamma，用“中心化设计矩阵”拟合，提升可辨识性；
      但最终返回时会把参数变换回题设“未中心化”公式，保证 P(t) 形式完全不变。

    关键点：中心化只用于拟合稳定性，不改变最终公式。
    变换关系：
      Ar*I_scr*(ps0' + ps1*(b - b_mean)) = Ar*I_scr*((ps0' - ps1*b_mean) + ps1*b)
      I_cpu*(pc0' + pc1*(u^g - u_mean)) = I_cpu*((pc0' - pc1*u_mean) + pc1*u^g)
    因此最终参数：
      ps0 = ps0' - ps1*b_mean
      pc0 = pc0' - pc1*u_mean
    """
    I_scr = df["screen_on"].to_numpy(dtype=float)
    b = df["brightness"].to_numpy(dtype=float)
    u = df["cpu_util"].to_numpy(dtype=float)
    I_cpu = (u > 1e-3).astype(float)

    I_wifi = (df["net_mode"].astype(str).str.lower() == "wifi").astype(float).to_numpy()
    I_4g = (df["net_mode"].astype(str).str.lower() == "4g").astype(float).to_numpy()
    I_5g = (df["net_mode"].astype(str).str.lower() == "5g").astype(float).to_numpy()

    I_gps = df["gps_on"].to_numpy(dtype=float)
    I_bg = df["bg_on"].to_numpy(dtype=float)
    a_bg = df["bg_intensity"].to_numpy(dtype=float)

    best_mse = float("inf")
    best = None

    for gamma in gamma_grid:
        ug = u ** float(gamma)

        # ---- 中心化均值（仅用于拟合稳定）----
        b_mean = float(np.mean(b[I_scr > 0.5])) if np.any(I_scr > 0.5) else 0.0
        u_mean = float(np.mean(ug[I_cpu > 0.5])) if np.any(I_cpu > 0.5) else 0.0
        b_center = b - b_mean
        u_center = ug - u_mean

        # 设计矩阵：对“中心化参数”做线性拟合
        # 参数向量：x = [p0, ps0', ps1, pc0', pc1, pwifi, p4g, p5g, pgps, pbg0, pbg1]
        A = np.vstack([
            np.ones_like(P_obs_W),          # p0
            area_ratio * I_scr,             # ps0'（中心化截距）
            area_ratio * I_scr * b_center,  # ps1
            I_cpu,                          # pc0'（中心化截距）
            I_cpu * u_center,               # pc1
            I_wifi,                         # pwifi
            I_4g,                           # p4g
            I_5g,                           # p5g
            I_gps,                          # pgps
            I_bg,                           # pbg0
            a_bg,                           # pbg1（题设为 + pbg1*a_bg，不额外乘I_bg）
        ]).T

        mask = np.isfinite(A).all(axis=1) & np.isfinite(P_obs_W)
        A_m = A[mask]
        y_m = P_obs_W[mask]

        lb = np.full(A_m.shape[1], eps, dtype=float)
        x = bounded_fit(A_m, y_m, lb=lb)

        resid = A_m @ x - y_m
        mse = float(np.mean(resid ** 2))

        if mse < best_mse:
            best_mse = mse
            best = (float(gamma), x, b_mean, u_mean)

    if best is None:
        raise RuntimeError("功耗参数拟合失败：请检查 P_obs_W 与特征列是否异常。")

    gamma_best, x_best, b_mean_best, u_mean_best = best

    # ---- 关键：把中心化参数变换回题设“未中心化”公式参数（保证最终P(t)不改）----
    p0 = float(x_best[0])

    ps0_prime = float(x_best[1])
    ps1 = float(x_best[2])
    ps0 = ps0_prime - ps1 * float(b_mean_best)

    pc0_prime = float(x_best[3])
    pc1 = float(x_best[4])
    pc0 = pc0_prime - pc1 * float(u_mean_best)

    return PowerParams(
        p0=p0,
        ps0=ps0,
        ps1=ps1,
        pc0=pc0,
        pc1=pc1,
        gamma=float(gamma_best),
        pwifi=float(x_best[5]),
        p4g=float(x_best[6]),
        p5g=float(x_best[7]),
        pgps=float(x_best[8]),
        pbg0=float(x_best[9]),
        pbg1=float(x_best[10]),
    )


# =========================
# 6) 用功耗参数预测 P_total(t)（严格题设公式）
# =========================

def predict_P_total(df: pd.DataFrame, pp: PowerParams, area_ratio: float) -> np.ndarray:
    """
    严格按题设功耗分解公式计算预测功耗（不允许改）：
    P = p0
      + Ar*I_scr*(ps0 + ps1*b)
      + I_cpu*(pc0 + pc1*u^gamma)
      + I_wifi*pwifi + I_4g*p4g + I_5g*p5g
      + I_gps*pgps
      + I_bg*pbg0
      + pbg1*a_bg
    """
    I_scr = df["screen_on"].to_numpy(dtype=float)
    b = df["brightness"].to_numpy(dtype=float)
    u = df["cpu_util"].to_numpy(dtype=float)
    I_cpu = (u > 1e-3).astype(float)

    ug = u ** pp.gamma

    I_wifi = (df["net_mode"].astype(str).str.lower() == "wifi").astype(float).to_numpy()
    I_4g = (df["net_mode"].astype(str).str.lower() == "4g").astype(float).to_numpy()
    I_5g = (df["net_mode"].astype(str).str.lower() == "5g").astype(float).to_numpy()

    I_gps = df["gps_on"].to_numpy(dtype=float)
    I_bg = df["bg_on"].to_numpy(dtype=float)
    a_bg = df["bg_intensity"].to_numpy(dtype=float)

    P = np.full_like(u, pp.p0, dtype=float)
    P += area_ratio * I_scr * (pp.ps0 + pp.ps1 * b)
    P += I_cpu * (pp.pc0 + pp.pc1 * ug)
    P += I_wifi * pp.pwifi + I_4g * pp.p4g + I_5g * pp.p5g
    P += I_gps * pp.pgps
    P += I_bg * pp.pbg0
    P += pp.pbg1 * a_bg  # 题设就是 + pbg1*a_bg，不额外乘 I_bg

    return np.maximum(P, 0.0)


# =========================
# 7) SOC 导数估计：滑动窗口线性回归
# =========================

def rolling_slope(time_s: np.ndarray, soc: np.ndarray, window: int = 31) -> np.ndarray:
    """
    对每个点 i，用窗口做线性回归：
      S(t) ≈ a t + b
    斜率 a 作为 dS/dt 的估计（SOC/秒）。
    """
    n = len(soc)
    dSdt = np.zeros(n, dtype=float)
    half = max(2, window // 2)

    for i in range(n):
        l = max(0, i - half)
        r = min(n, i + half + 1)
        tt = time_s[l:r] - time_s[l]
        yy = soc[l:r]
        if len(tt) < 3:
            continue
        A = np.vstack([tt, np.ones_like(tt)]).T
        slope, _ = np.linalg.lstsq(A, yy, rcond=None)[0]
        dSdt[i] = slope

    return dSdt


# =========================
# 8) 拟合温度参数：T_ref, α^-,β^-,α^+,β^+
# =========================

def fit_temp_params_dual(
    df: pd.DataFrame,
    P_total: np.ndarray,
    C0_Wh: float,
    k_age: float,
    eps: float = 1e-10
) -> TempParams:
    """
    严格依据题设双侧温度模型：
    ΔT^-=(T_ref - T(t))_+ ; ΔT^+=(T(t) - T_ref)_+
    dS/dt = -(P/(C0*k_age))*(1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+)

    与旧版一致的做法：通过 dS/dt 估计得到等效 y（W）：
      y = -(dS/dt)*3600*C_eff  , C_eff = C0*k_age
    理论上：
      y ≈ P*(1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+)
    移项：
      Z = y - P ≈ α^-*(PΔT^-) + β^-*(P^2ΔT^-)
                + α^+*(PΔT^+) + β^+*(P^2ΔT^+)

    对每个 T_ref 做网格搜索，并用带下界的最小二乘拟合 [α^-,β^-,α^+,β^+]。
    """
    t = df["time_s"].to_numpy(dtype=float)
    soc = df["soc"].to_numpy(dtype=float)

    # 题设中的 T(t) ：使用算法得到的 temp_opt_c（不是原始 temp_c）
    Topt = df["temp_opt_c"].to_numpy(dtype=float)

    dSdt = rolling_slope(t, soc, window=31)
    C_eff = C0_Wh * k_age
    y = -(dSdt * 3600.0) * C_eff  # 单位W

    # 过滤：y>0 避免导数噪声导致的非物理点
    mask = np.isfinite(y) & np.isfinite(P_total) & np.isfinite(Topt) & (y > 0) & (P_total > 0)
    y = y[mask]
    P = P_total[mask]
    Tm = Topt[mask]

    Tmin = float(np.min(Tm))
    Tmax = float(np.max(Tm))
    T_ref_grid = np.arange(Tmin - 2.0, Tmax + 2.0, 0.5)

    best_mse = float("inf")
    best = None

    for T_ref in T_ref_grid:
        dT_minus = pos_part(float(T_ref) - Tm)   # ΔT^-
        dT_plus = pos_part(Tm - float(T_ref))    # ΔT^+

        Z = y - P

        X1 = P * dT_minus
        X2 = (P ** 2) * dT_minus
        X3 = P * dT_plus
        X4 = (P ** 2) * dT_plus

        A = np.vstack([X1, X2, X3, X4]).T

        lb = np.full(A.shape[1], eps, dtype=float)  # α/β 一般取非负更符合“惩罚”语义
        ab = bounded_fit(A, Z, lb=lb)

        resid = A @ ab - Z
        mse = float(np.mean(resid ** 2))

        if mse < best_mse:
            best_mse = mse
            best = (float(T_ref), float(ab[0]), float(ab[1]), float(ab[2]), float(ab[3]))

    if best is None:
        raise RuntimeError("温度参数拟合失败：请检查SOC变化、温度范围或功耗序列。")

    return TempParams(
        T_ref=best[0],
        alpha_minus=best[1],
        beta_minus=best[2],
        alpha_plus=best[3],
        beta_plus=best[4],
    )


# =========================
# 9) SOC 仿真：欧拉离散积分（严格题设微分方程）
# =========================

def simulate_soc_dual_temp(
    df: pd.DataFrame,
    P_total: np.ndarray,
    tp: TempParams,
    C0_Wh: float,
    k_age: float
) -> np.ndarray:
    """
    严格离散题设微分方程：
      dS/dt = -(P/(C0*k_age))*(1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+)

    欧拉离散：
      S_i = max(0, S_{i-1} - P_i*eta_i*dt_h/(C0*k_age))
    其中：
      eta_i = 1 + α^-ΔT^- + β^-PΔT^- + α^+ΔT^+ + β^+PΔT^+
      dt_h = dt_s / 3600
    """
    t = df["time_s"].to_numpy(dtype=float)
    dt_s = np.diff(t, prepend=t[0])
    if len(t) > 1:
        dt_s[0] = float(np.median(np.diff(t)))
    dt_h = dt_s / 3600.0

    # T(t)：temp_opt_c
    Topt = df["temp_opt_c"].to_numpy(dtype=float)
    dT_minus = pos_part(tp.T_ref - Topt)
    dT_plus = pos_part(Topt - tp.T_ref)

    eta = (
        1.0
        + tp.alpha_minus * dT_minus
        + tp.beta_minus * P_total * dT_minus
        + tp.alpha_plus * dT_plus
        + tp.beta_plus * P_total * dT_plus
    )

    denom = C0_Wh * k_age
    S = np.empty_like(P_total, dtype=float)
    S[0] = float(df["soc"].iloc[0])

    for i in range(1, len(S)):
        dS = (P_total[i] * eta[i] * dt_h[i]) / denom
        S[i] = max(0.0, S[i - 1] - dS)

    return S


# =========================
# 10) 可视化与诊断
# =========================

def compute_power_components(df: pd.DataFrame, pp: PowerParams, area_ratio: float) -> pd.DataFrame:
    """
    将预测功耗拆成多个分量，便于堆叠图展示“功耗归因”。
    注意：分量计算严格对应题设功率公式。
    """
    I_scr = df["screen_on"].to_numpy(dtype=float)
    b = df["brightness"].to_numpy(dtype=float)
    u = df["cpu_util"].to_numpy(dtype=float)
    I_cpu = (u > 1e-3).astype(float)
    ug = u ** pp.gamma

    I_wifi = (df["net_mode"].astype(str).str.lower() == "wifi").astype(float).to_numpy()
    I_4g = (df["net_mode"].astype(str).str.lower() == "4g").astype(float).to_numpy()
    I_5g = (df["net_mode"].astype(str).str.lower() == "5g").astype(float).to_numpy()

    I_gps = df["gps_on"].to_numpy(dtype=float)
    I_bg = df["bg_on"].to_numpy(dtype=float)
    a_bg = df["bg_intensity"].to_numpy(dtype=float)

    comps = pd.DataFrame()
    comps["P_base"] = np.full_like(u, pp.p0, dtype=float)
    comps["P_scr"] = area_ratio * I_scr * (pp.ps0 + pp.ps1 * b)
    comps["P_cpu"] = I_cpu * (pp.pc0 + pp.pc1 * ug)
    comps["P_wifi"] = I_wifi * pp.pwifi
    comps["P_4g"] = I_4g * pp.p4g
    comps["P_5g"] = I_5g * pp.p5g
    comps["P_gps"] = I_gps * pp.pgps
    comps["P_bg_sw"] = I_bg * pp.pbg0
    comps["P_bg_int"] = pp.pbg1 * a_bg
    comps["P_total"] = comps.sum(axis=1).to_numpy()
    return comps


def diagnose_gamma_search(
    df: pd.DataFrame,
    P_obs_W: np.ndarray,
    gamma_grid: np.ndarray,
    area_ratio: float,
    eps: float = 1e-6
) -> pd.DataFrame:
    """
    记录 gamma 网格搜索的误差与（最终公式）参数值，便于论文展示“gamma如何确定”。
    做法：与 fit_power_params_from_Pobs 一致（中心化拟合 + 变换回最终公式参数）。
    """
    I_scr = df["screen_on"].to_numpy(dtype=float)
    b = df["brightness"].to_numpy(dtype=float)
    u = df["cpu_util"].to_numpy(dtype=float)
    I_cpu = (u > 1e-3).astype(float)

    I_wifi = (df["net_mode"].astype(str).str.lower() == "wifi").astype(float).to_numpy()
    I_4g = (df["net_mode"].astype(str).str.lower() == "4g").astype(float).to_numpy()
    I_5g = (df["net_mode"].astype(str).str.lower() == "5g").astype(float).to_numpy()
    I_gps = df["gps_on"].to_numpy(dtype=float)
    I_bg = df["bg_on"].to_numpy(dtype=float)
    a_bg = df["bg_intensity"].to_numpy(dtype=float)

    rows = []
    for gamma in gamma_grid:
        ug = u ** float(gamma)
        b_mean = float(np.mean(b[I_scr > 0.5])) if np.any(I_scr > 0.5) else 0.0
        u_mean = float(np.mean(ug[I_cpu > 0.5])) if np.any(I_cpu > 0.5) else 0.0
        b_center = b - b_mean
        u_center = ug - u_mean

        A = np.vstack([
            np.ones_like(P_obs_W),
            area_ratio * I_scr,
            area_ratio * I_scr * b_center,
            I_cpu,
            I_cpu * u_center,
            I_wifi,
            I_4g,
            I_5g,
            I_gps,
            I_bg,
            a_bg,
        ]).T

        mask = np.isfinite(A).all(axis=1) & np.isfinite(P_obs_W)
        A_m = A[mask]
        y_m = P_obs_W[mask]
        lb = np.full(A_m.shape[1], eps, dtype=float)
        x = bounded_fit(A_m, y_m, lb=lb)

        resid = A_m @ x - y_m
        mse = float(np.mean(resid ** 2))

        # 变换回最终公式参数
        ps0 = float(x[1]) - float(x[2]) * b_mean
        ps1 = float(x[2])
        pc0 = float(x[3]) - float(x[4]) * u_mean
        pc1 = float(x[4])

        rows.append({
            "gamma": float(gamma),
            "mse": mse,
            "p0": float(x[0]),
            "ps0": ps0,
            "ps1": ps1,
            "pc0": pc0,
            "pc1": pc1,
            "pwifi": float(x[5]),
            "p4g": float(x[6]),
            "p5g": float(x[7]),
            "pgps": float(x[8]),
            "pbg0": float(x[9]),
            "pbg1": float(x[10]),
            "b_mean_used_for_fit": float(b_mean),
            "u_mean_used_for_fit": float(u_mean),
        })

    return pd.DataFrame(rows)


def diagnose_Tref_search_dual(
    df: pd.DataFrame,
    P_total: np.ndarray,
    C0_Wh: float,
    k_age: float,
    eps: float = 1e-10
) -> pd.DataFrame:
    """
    复刻双侧温度拟合中的 T_ref 网格搜索过程，把每个 T_ref 的 MSE 与参数记录下来。
    """
    t = df["time_s"].to_numpy(dtype=float)
    soc = df["soc"].to_numpy(dtype=float)
    Topt = df["temp_opt_c"].to_numpy(dtype=float)

    dSdt = rolling_slope(t, soc, window=31)
    y = -(dSdt * 3600.0) * (C0_Wh * k_age)

    mask = np.isfinite(y) & np.isfinite(P_total) & np.isfinite(Topt) & (y > 0) & (P_total > 0)
    y = y[mask]
    P = P_total[mask]
    Tm = Topt[mask]

    Tmin = float(np.min(Tm))
    Tmax = float(np.max(Tm))
    T_ref_grid = np.arange(Tmin - 2.0, Tmax + 2.0, 0.5)

    rows = []
    for T_ref in T_ref_grid:
        dT_minus = pos_part(float(T_ref) - Tm)
        dT_plus = pos_part(Tm - float(T_ref))

        Z = y - P
        X1 = P * dT_minus
        X2 = (P ** 2) * dT_minus
        X3 = P * dT_plus
        X4 = (P ** 2) * dT_plus
        A = np.vstack([X1, X2, X3, X4]).T

        lb = np.full(A.shape[1], eps, dtype=float)
        ab = bounded_fit(A, Z, lb=lb)

        resid = A @ ab - Z
        mse = float(np.mean(resid ** 2))

        rows.append({
            "T_ref": float(T_ref),
            "mse": mse,
            "alpha_minus": float(ab[0]),
            "beta_minus": float(ab[1]),
            "alpha_plus": float(ab[2]),
            "beta_plus": float(ab[3]),
        })

    return pd.DataFrame(rows)


def plot_parameter_determination(
    df: pd.DataFrame,
    P_obs_W: np.ndarray,
    pp: PowerParams,
    P_model_W: np.ndarray,
    tp: TempParams,
    C0_Wh: float,
    k_age: float,
    area_ratio: float,
    gamma_grid: np.ndarray
) -> None:
    """
    统一出图入口：
      - fig_gamma_mse.png：gamma网格搜索MSE曲线
      - fig_Tref_mse.png：T_ref网格搜索MSE曲线（双侧温度）
      - fig_power_scatter.png：P_obs vs P_model
      - fig_power_residual_time.png：残差随时间
      - fig_power_components_stack.png：功耗分量堆叠
      - fig_eta_time.png：温度惩罚因子 eta(t)
      - diag_gamma_search.csv / diag_Tref_search.csv：诊断表导出
    """
    gamma_diag = diagnose_gamma_search(df, P_obs_W, gamma_grid, area_ratio=area_ratio, eps=1e-6)
    Tref_diag = diagnose_Tref_search_dual(df, P_model_W, C0_Wh=C0_Wh, k_age=k_age, eps=1e-10)

    gamma_diag.to_csv("diag_gamma_search.csv", index=False)
    Tref_diag.to_csv("diag_Tref_search.csv", index=False)

    # 图1：gamma vs MSE
    plt.figure()
    plt.plot(gamma_diag["gamma"].to_numpy(), gamma_diag["mse"].to_numpy())
    plt.xlabel("gamma")
    plt.ylabel("MSE (power fit)")
    plt.title("Gamma grid search: MSE vs gamma")
    plt.axvline(pp.gamma, linestyle="--")
    plt.tight_layout()
    plt.savefig("fig_gamma_mse.png", dpi=200)
    plt.close()

    # 图2：T_ref vs MSE
    plt.figure()
    plt.plot(Tref_diag["T_ref"].to_numpy(), Tref_diag["mse"].to_numpy())
    plt.xlabel("T_ref (°C)")
    plt.ylabel("MSE (temp fit, dual-sided)")
    plt.title("T_ref grid search: MSE vs T_ref (dual-sided)")
    plt.axvline(tp.T_ref, linestyle="--")
    plt.tight_layout()
    plt.savefig("fig_Tref_mse.png", dpi=200)
    plt.close()

    # 图3：P_obs vs P_model
    plt.figure()
    plt.scatter(P_obs_W, P_model_W, s=8)
    lim0 = float(min(np.min(P_obs_W), np.min(P_model_W)))
    lim1 = float(max(np.max(P_obs_W), np.max(P_model_W)))
    plt.plot([lim0, lim1], [lim0, lim1])
    plt.xlabel("P_obs (W)")
    plt.ylabel("P_model (W)")
    plt.title("Power fit: P_obs vs P_model")
    plt.tight_layout()
    plt.savefig("fig_power_scatter.png", dpi=200)
    plt.close()

    # 图4：残差随时间
    plt.figure()
    plt.plot(df["time_s"].to_numpy(), (P_obs_W - P_model_W))
    plt.xlabel("time (s)")
    plt.ylabel("P_obs - P_model (W)")
    plt.title("Power residual over time")
    plt.tight_layout()
    plt.savefig("fig_power_residual_time.png", dpi=200)
    plt.close()

    # 图5：功耗分量堆叠
    comps = compute_power_components(df, pp, area_ratio=area_ratio)
    t = df["time_s"].to_numpy(dtype=float)
    plt.figure()
    stack_cols = ["P_base", "P_scr", "P_cpu", "P_wifi", "P_4g", "P_5g", "P_gps", "P_bg_sw", "P_bg_int"]
    plt.stackplot(t, [comps[c].to_numpy() for c in stack_cols], labels=stack_cols)
    plt.xlabel("time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power components (stacked)")
    plt.legend(loc="upper right", fontsize=7)
    plt.tight_layout()
    plt.savefig("fig_power_components_stack.png", dpi=200)
    plt.close()

    # 图6：温度惩罚因子 eta(t)（严格题设形式）
    Topt = df["temp_opt_c"].to_numpy(dtype=float)
    dT_minus = pos_part(tp.T_ref - Topt)
    dT_plus = pos_part(Topt - tp.T_ref)
    eta = (
        1.0
        + tp.alpha_minus * dT_minus
        + tp.beta_minus * P_model_W * dT_minus
        + tp.alpha_plus * dT_plus
        + tp.beta_plus * P_model_W * dT_plus
    )
    plt.figure()
    plt.plot(t, eta)
    plt.xlabel("time (s)")
    plt.ylabel("eta(t)")
    plt.title("Temperature penalty factor eta(t) (dual-sided)")
    plt.tight_layout()
    plt.savefig("fig_eta_time.png", dpi=200)
    plt.close()


# =========================
# 11) 参数保存/读取（用于复现与图表展示）
# =========================

def save_params_json(pp: PowerParams, tp: TempParams, path: str = "fitted_params.json") -> None:
    """把拟合得到的 pp,tp 保存为 json，便于复现与后续出图。"""
    payload = {
        "PowerParams": asdict(pp),
        "TempParams": asdict(tp),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_params_json(path: str = "fitted_params.json") -> typing.Tuple[PowerParams, TempParams]:
    """从 json 读取 pp,tp（如果你只想出图，不想重跑拟合）。"""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pp = PowerParams(**payload["PowerParams"])
    tp = TempParams(**payload["TempParams"])
    return pp, tp


# =========================
# 12) 主流程：拟合 + 仿真 + 导出 + 可视化
# =========================

def main():
    # 新数据路径（你已上传到该位置）
    data_path = "battery_dataset.csv"

    # 电池标称可用能量（Wh）：按机型修改
    C0_Wh = 15.0
    # 老化系数（无量纲）：若不考虑老化取1
    k_age = 1.0

    # 屏幕面积比 Ar = A_screen / A_ref（相对量即可）
    A_screen = 1.0
    A_ref = 1.0
    area_ratio = float(A_screen / A_ref)

    # 读取并构造特征 + 观测功耗（适配新CSV）
    df, P_obs_W = build_model_df_from_battery_dataset(data_path, choose_device=None)

    # gamma 搜索网格
    gamma_grid = np.linspace(1.0, 3.0, 21)

    # 1) 拟合功耗参数（最终参数严格对应题设P(t)公式）
    pp = fit_power_params_from_Pobs(df, P_obs_W, gamma_grid=gamma_grid, area_ratio=area_ratio, eps=1e-6)

    # 2) 预测功耗序列
    P_model_W = predict_P_total(df, pp, area_ratio=area_ratio)

    # 3) 拟合双侧温度参数（严格题设SOC微分方程形式）
    tp = fit_temp_params_dual(df, P_model_W, C0_Wh=C0_Wh, k_age=k_age, eps=1e-10)

    # 4) SOC 仿真（双侧温度）
    S_pred = simulate_soc_dual_temp(df, P_model_W, tp, C0_Wh=C0_Wh, k_age=k_age)

    # 导出结果表
    out = df.copy()
    out["P_obs_W"] = P_obs_W
    out["P_model_W"] = P_model_W
    out["S_pred"] = S_pred
    out.to_csv("fitted_model_output.csv", index=False)

    # 保存参数
    save_params_json(pp, tp, path="fitted_params.json")

    # 画图 + 导出诊断表
    plot_parameter_determination(
        df=df,
        P_obs_W=P_obs_W,
        pp=pp,
        P_model_W=P_model_W,
        tp=tp,
        C0_Wh=C0_Wh,
        k_age=k_age,
        area_ratio=area_ratio,
        gamma_grid=gamma_grid
    )

    print("===== PowerParams（严格题设P(t)参数）=====")
    print(pp)
    print("===== TempParams（双侧温度惩罚参数）=====")
    print(tp)
    print("已输出：fitted_model_output.csv")
    print("已输出：fitted_params.json")
    print("已输出图表：fig_*.png 与诊断表：diag_*.csv")


if __name__ == "__main__":
    main()
