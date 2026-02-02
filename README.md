# 2026MCM-A: Smartphone Battery Life Modeling and Analysis

Users are constantly plagued by the unpredictability of smartphone battery life: the same device might last a full day one time, yet run out before lunchtime the next. To better understand these fluctuations, we built a **continuous-time, temperature-coupled State of Charge (SOC) framework** based on publicly available smartphone test datasets. This framework analyzes how the screen, CPU, wireless networks (Wi-Fi/4G/5G), GPS, background services, and ambient temperature drive SOC changes across different usage scenarios.

## 📊 Data Processing & Visualization
1.  **Data Preprocessing**: Multi-source time series data were time-aligned, with duplicates and outliers removed.
2.  **Noise Smoothing**: High-frequency noise was smoothed using short-window filtering.
3.  **Data Normalization**: Processed data were standardized.
4.  **Visual Analysis**: Correlation heatmaps, bin-based contour plots, and 3D surface plots were created to reveal relationships. Analysis indicated that **screen display** and **computational load** are the primary drivers of system energy consumption.

---

## 🔧 Model Construction

### Model 1: Power Consumption Mechanism Model (Mechanism Layer)
*   **Total Power Representation**: Modeled as the sum of seven key component metrics.
*   **Parameter Estimation**: **Constrained Least Squares Estimation** was employed.
*   **Non-linear CPU Term**: An exponential factor (γ) was determined via grid search, ultimately set to **γ = 3.0** to minimize fitting error.
*   **Key Parameter Examples**:
    *   Wireless network coefficients show an increasing trend:
        `p_WiFi = 0.0666`, `p_4G = 0.0991`, `p_5G = 0.1077`
    *   High-power CPU term:
        `p_c1 = 1.3149`

### Model 2: Physics-Coupled Model (Physical Layer)
*   **Temperature Effect Modeling**: Separately modeled the impact of temperatures above and below a reference point on the power dissipation rate.
*   **Reference Temperature**: Determined via grid search to be **16.7981°C**; other coefficients were fitted using constrained regression.
*   **SOC Dynamics**: A first-order Ordinary Differential Equation (ODE) for SOC was established based on the total power model and temperature effects.
*   **Numerical Solution**:
    *   SOC was advanced using the **Explicit Euler method**.
    *   **Non-negativity constraints** ensured physical feasibility.
    *   Depletion time was refined via **linear interpolation** when SOC crossed zero.
*   **Time-To-Empty (TTE) Calculation**: An efficient TTE calculation method combining discrete cumulative summation and interpolation was proposed, generating the **SOC–TTE curve**.

---

## 📈 Scenario Analysis & Application

### Typical Usage Scenario Simulation
Based on the established model, **ten representative usage scenarios** were constructed. For each:
*   TTE and component power share were calculated.
*   **Key Findings**:
    *   "Rapid power-down" is primarily driven by **5G communication** and **GPS positioning**.
    *   **Background services** exhibit significant cumulative impact in low-interaction scenarios (e.g., music playback, standby).

### Parameter Sensitivity Analysis
*   **Method**: **One-at-a-Time (OAT) sensitivity analysis** was performed by applying negative perturbations to parameters sequentially.
*   **Metric**: Parameters were ranked using **ΔP_eff** to identify those requiring priority calibration.
*   **Conclusion**: The model demonstrates strong **accuracy** and **robustness** across diverse scenarios.

---

## 🔄 Battery Aging Modeling
*   **Aging Factor**: Introduced **exponential aging factors** correlated with calendar time and cumulative energy flux.
*   **Aging Scenario Curves**: Simulation shows that more severe aging **systematically reduces endurance** at the same SOC.
*   **Implication**: Runtime TTE estimation and control thresholds should **adaptively update with the battery's aging state**.

---

> This project systematically uncovers the key drivers behind smartphone battery life fluctuations through a mechanism-and-data-fused modeling approach. It provides a theoretical foundation and practical toolkit for dynamic SOC prediction and aging-adaptive power management.
