# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 10:32:28 2025

@author: luoh
"""
# Improved CNS-style plotting function for bacterial growth with full pipeline + Excel export
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit, fsolve
import os
import pandas as pd

# CNS-style aesthetic settings
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 14
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.size'] = 6
rcParams['xtick.major.width'] = 1.2
rcParams['ytick.major.size'] = 6
rcParams['ytick.major.width'] = 1.2
rcParams['legend.frameon'] = False
rcParams['axes.grid'] = False
rcParams['savefig.transparent'] = True

from matplotlib import cm
COLOR_MAP = plt.cm.tab10(np.linspace(0, 1, 10))

def logistic(t, K, N0, r):
    return K / (1 + ((K - N0) / N0) * np.exp(-r * t))

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 1.0

def find_intersection_time(K, N0, r, target):
    def equation(t):
        return logistic(t, K, N0, r) - target
    try:
        term = (N0 * (K - target)) / ((K - N0) * target)
        if term <= 0:
            return None
        t_guess = -np.log(term) / r
        t_solution = fsolve(equation, t_guess)[0]
        if 0 <= t_solution <= 25 and abs(equation(t_solution)) < 1e-6:
            return t_solution
    except:
        return None
    return None

def plot_growth_CNS(time_points, mean_cfu, std_cfu, fit_params, r2_vals, intersection_times, target_y, output_file):
    fig, ax = plt.subplots(figsize=(8, 6))
    t_fit = np.linspace(0, 25, 200)

    for idx, name in enumerate(mean_cfu.keys()):
        color = COLOR_MAP[idx % len(COLOR_MAP)]
        y_mean = mean_cfu[name]
        y_std = std_cfu[name]
        popt = fit_params[name]
        t_int = intersection_times[name]

        ax.errorbar(time_points, y_mean, yerr=y_std, fmt='o', ms=5, lw=1.5, capsize=3, color=color, label=f'{name}')
        if popt is not None:
            y_fit = logistic(t_fit, *popt)
            ax.plot(t_fit, y_fit, '-', lw=2, color=color)
            if t_int is not None:
                ax.plot(t_int, target_y, 'v', color=color, markersize=7)
                ax.text(t_int, target_y * 1.3, f'{t_int:.1f}h', color=color, fontsize=12, ha='center', va='bottom')

    ax.axhline(y=target_y, color='gray', linestyle='--', lw=1.2)
    ax.text(5.5, target_y * 1.05, f"{target_y:.0e} CFU/mL", ha='right', fontsize=13, color='gray')

    ax.set_xscale('linear')
    ax.set_yscale('log')
    ax.set_xlim(0, 25)
    ax.set_ylim(1e5, 1e9)
    ax.set_xlabel('Time (h)', fontsize=20, fontweight='bold')
    ax.set_ylabel('Bacterial Concentration (CFU/mL)', fontsize=20, fontweight='bold')
    plt.xticks (fontweight='bold', fontsize=16)
    plt.yticks (fontweight='bold', fontsize=16)
    ax.legend(loc='upper left', fontsize=12, ncol=2)
    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.show()
    print(f"Saved to {os.path.abspath(output_file)}")

def export_results_to_excel(fit_params, r2_vals, intersection_times, output_excel):
    rows = []
    for name in fit_params:
        K, N0, r = fit_params[name] if fit_params[name] is not None else (None, None, None)
        R2 = r2_vals.get(name)
        Tint = intersection_times.get(name)
        rows.append({'Sample': name, 'K': K, 'N0': N0, 'r': r, 'R²': R2, 'Intersection Time (h)': Tint})
    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"Exported results to {output_excel}")

def analyze_growth(data, sample_names, time_points, target_y, conversion_factor, output_file, output_excel):
    mean_cfu, std_cfu, fit_params, r2_vals, intersection_times = {}, {}, {}, {}, {}

    for name in sample_names:
        data[name] = np.maximum(data[name], 0) * conversion_factor
        mean_cfu[name] = np.mean(data[name], axis=0)
        std_cfu[name] = np.std(data[name], axis=0)

    bounds = ([1e7, 1e5, 0], [1e9, 1e8, 1])
    for name in sample_names:
        try:
            popt, _ = curve_fit(logistic, time_points, mean_cfu[name],
                                p0=[5e8, 1e6, 0.3], bounds=bounds)
            fit_params[name] = popt
            r2_vals[name] = r_squared(mean_cfu[name], logistic(time_points, *popt))
            intersection_times[name] = find_intersection_time(*popt, target_y)
        except RuntimeError:
            fit_params[name] = None
            r2_vals[name] = None
            intersection_times[name] = None

    plot_growth_CNS(time_points, mean_cfu, std_cfu, fit_params, r2_vals, intersection_times, target_y, output_file)
    export_results_to_excel(fit_params, r2_vals, intersection_times, output_excel)

# Sample usage
if __name__ == '__main__':
    data = {
        '1e1': np.array([[-0.013, -0.006, 0, -0.004, -0.004, -0.002, 0.004, 0.046, 0.366],
                        [-0.018, -0.007, -0.003, -0.006, -0.005, -0.003, 0.011, 0.076, 0.436],
                        [-0.013, -0.002, -0.005, 0, 0.001, 0.009, 0.031, 0.107, 0.378],
                        [-0.018, 0, 0.001, -0.002, 0, 0.006, 0.02, 0.111, 0.386]]),
        '1e2': np.array([[-0.023, -0.008, -0.005, -0.002, 0.017, 0.118, 0.101, 0.183, 0.246],
                        [-0.019, -0.007, -0.006, 0.004, 0.019, 0.095, 0.164, 0.212, 0.364],
                        [-0.021, -0.007, -0.002, 0.005, 0.013, 0.112, 0.221, 0.167, 0.352],
                        [-0.022, 0.001, -0.002, -0.002, 0.018, 0.087, 0.237, 0.26, 0.279]]),
        '1e3': np.array([[-0.02, -0.001, 0.013, 0.112, 0.111, 0.213, 0.236, 0.3, 0.329],
                        [-0.016, -0.002, 0.009, 0.069, 0.069, 0.292, 0.281, 0.296, 0.326],
                        [-0.015, -0.004, 0.014, 0.089, 0.095, 0.231, 0.31, 0.325, 0.316],
                        [-0.009, 0, 0.015, 0.08, 0.107, 0.315, 0.26, 0.252, 0.304]]),
        '1e4': np.array([[0.027, 0.047, 0.086, 0.135, 0.248, 0.362, 0.429, 0.442, 0.443],
                        [0.023, 0.043, 0.074, 0.117, 0.209, 0.331, 0.372, 0.47, 0.43],
                        [0.021, 0.046, 0.082, 0.123, 0.227, 0.345, 0.4, 0.491, 0.444],
                        [0.032, 0.048, 0.083, 0.136, 0.239, 0.36, 0.41, 0.43, 0.43]])
    }
    sample_names = ['1e1', '1e2', '1e3', '1e4']
    time_points = np.array([0, 2, 4, 6, 8, 10, 12, 14, 24])
    target_y = 2e8
    conversion_factor = 8e8
    output_file = "bacterial_growth_CNS-3.png"
    output_excel = "growth_curve_fitting_results.xlsx"
    analyze_growth(data, sample_names, time_points, target_y, conversion_factor, output_file, output_excel)