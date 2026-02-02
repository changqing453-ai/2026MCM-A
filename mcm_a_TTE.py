import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义模型的PowerParams和TempParams
PARAMS = {
    "PowerParams": {
        "p0": 0.12523029046649736,
        "ps0": -0.09581395111117268,
        "ps1": 0.8450015788758214,
        "pc0": 0.036985840269651495,
        "pc1": 1.9101423809027038,
        "gamma": 3.0,
        "pwifi": -0.027996651322704798,
        "p4g": 0.2319508717559768,
        "p5g": 0.2278152637582298,
        "pgps": 0.4051127613896484,
        "pbg0": 0.055365670536850535,
        "pbg1": 0.2757823883595569
    },
    "TempParams": {
        "T_ref": 25.60723829078337,
        "alpha_minus": 0.029921386418753835,
        "beta_minus": 2.9935041840913747e-06,
        "alpha_plus": 6.338633922765952e-07,
        "beta_plus": 0.010036402208481095
    }
}

# 控制参数
SCENARIO = {
    "screen_on": 1,
    "brightness": 0.80,
    "cpu_util": 0.85,
    "net_mode": "wifi",
    "gps_on": 0,
    "bg_on": 0,
    "bg_intensity": 0.00,
    "temp_c": 44.7
}

AREA_RATIO = 1.0

# 计算功率P
def compute_power(controls, params, area_ratio=1.0):
    pp = params["PowerParams"]
    b = float(np.clip(controls["brightness"], 0.0, 1.0))
    u = float(np.clip(controls["cpu_util"], 0.0, 1.0))

    I_scr = 1.0 if float(controls["screen_on"]) >= 0.5 else 0.0
    I_cpu = 1.0 if u > 1e-3 else 0.0
    I_gps = 1.0 if float(controls["gps_on"]) >= 0.5 else 0.0
    I_bg = 1.0 if float(controls["bg_on"]) >= 0.5 else 0.0

    nm = (controls["net_mode"] or "none").strip().lower()
    I_wifi = 1.0 if nm == "wifi" else 0.0
    I_4g = 1.0 if nm == "4g" else 0.0
    I_5g = 1.0 if nm == "5g" else 0.0

    a_bg = float(np.clip(controls.get("bg_intensity", 0.0), 0.0, 1.0))

    P = (
        pp["p0"]
        + area_ratio * I_scr * (pp["ps0"] + pp["ps1"] * b)
        + I_cpu * (pp["pc0"] + pp["pc1"] * (u ** pp["gamma"]))
        + I_wifi * pp["pwifi"] + I_4g * pp["p4g"] + I_5g * pp["p5g"]
        + I_gps * pp["pgps"]
        + I_bg * pp["pbg0"]
        + pp["pbg1"] * a_bg
    )
    return P

# 计算温度相关项
def compute_phi(P, T_eff, params):
    tp = params["TempParams"]
    T_ref = float(tp["T_ref"])

    dT_minus = max(T_ref - T_eff, 0.0)
    dT_plus = max(T_eff - T_ref, 0.0)

    Phi = (
        1.0
        + tp["alpha_minus"] * dT_minus + tp["beta_minus"] * P * dT_minus
        + tp["alpha_plus"] * dT_plus + tp["beta_plus"] * P * dT_plus
    )
    return {"T_ref": T_ref, "dT_minus": float(dT_minus), "dT_plus": float(dT_plus), "Phi": float(Phi)}

# 反向积分求TTE
def inverse_time_by_discrete_cumsum(S0, S_target, dt, r, t_max=1e7):
    need = max(S0 - S_target, 0.0)
    if need <= 0:
        return 0.0
    if r <= 0:
        return np.inf

    dG = r * dt
    if dG <= 0:
        return np.inf

    n = int(np.ceil(need / dG))
    if n * dt > t_max:
        return np.inf

    G_nm1 = (n - 1) * dG
    G_n = n * dG

    t = (n - 1) * dt + dt * (need - G_nm1) / (G_n - G_nm1)
    return float(t)

# 计算SOC与TTE的关系
def soc_sweep_tte(SOC_list, scenario, params, area_ratio, S_min=0.01, dt=1.0, C0=1.0, k_age=1.0):
    soc_arr = np.asarray(SOC_list, dtype=float)

    P = compute_power(scenario, params, area_ratio=area_ratio)
    mid = compute_phi(P=P, T_eff=float(scenario["temp_c"]), params=params)

    Phi = mid["Phi"]
    r = (P / (C0 * k_age)) * Phi

    rows = []
    for S0 in soc_arr:
        tte = inverse_time_by_discrete_cumsum(S0=float(S0), S_target=float(S_min), dt=float(dt), r=float(r))
        rows.append({
            "SOC0": float(S0),
            "S_min": float(S_min),
            "TTE": float(tte),
            "P": float(P),
            "T_eff": float(scenario["temp_c"]),
            "T_ref": float(mid["T_ref"]),
            "dT_minus": float(mid["dT_minus"]),
            "dT_plus": float(mid["dT_plus"]),
            "Phi": float(Phi),
            "r": float(r),
            "dt": float(dt),
            "C0": float(C0),
            "k_age": float(k_age),
        })

    df = pd.DataFrame(rows).sort_values("SOC0").reset_index(drop=True)
    return df

# 绘制SOC-TTE曲线
def plot_soc_tte(df, title):
    plt.figure(figsize=(8, 6))
    plt.plot(df["SOC0"].to_numpy(), df["TTE"].to_numpy(), marker="o", color='b')
    plt.xlabel("SOC (%)")
    plt.ylabel("Time-to-Empty (TTE) (hours)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 主程序：跑不同kage（1.0，0.9，0.8）并作图
if __name__ == "__main__":
    soc_list = np.linspace(0.05, 1.0, 20)

    kage_values = [1.0, 1.2, 1.4,1.6,1.8]
    plt.figure(figsize=(8, 6))

    for kage in kage_values:
        df = soc_sweep_tte(
            soc_list,
            scenario=SCENARIO,
            params=PARAMS,
            area_ratio=AREA_RATIO,
            S_min=0.01,
            dt=1.0,
            C0=1.0,
            k_age=kage
        )
        plt.plot(df["SOC0"], df["TTE"], label=f'kage={kage}')

    plt.xlabel("SOC (%)")
    plt.ylabel("Time-to-Empty (TTE) (hours)")
    plt.title("SOC vs Time-to-Empty (TTE) for Different Aging Factors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
