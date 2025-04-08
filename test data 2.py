import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root   
from scipy.integrate import solve_ivp
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# Define creep rate function
def creep_rate(t, y, beta, delta, zeta, epsilon=1e-6):
    return beta / ((delta * t + zeta + epsilon) * np.log(delta * t + zeta + epsilon))

# Define material data in a dictionary for better referencing
creep_data = {
       240: {"Ec_0": 10.01e-4, "Ec_m": 5e-5, "t_m": 100, "t_f": 23e2, "file": "combined_2.csv", "strain_col": "240strainoffset", "time_col": "240timeoffset"},
       280: {"Ec_0": 16.5e-3, "Ec_m": 10e-5, "t_m": 80, "t_f": 19e2, "file": "combined_2.csv", "strain_col": "280strainoffset", "time_col": "280timeoffset"},    
       300: {"Ec_0": 27e-3, "Ec_m": 20e-5, "t_m": 70, "t_f": 8e2, "file": "combined_2.csv", "strain_col": "300strainoffset", "time_col": "300timeoffset"},
       320: {"Ec_0": 40e-3, "Ec_m": 40e-5, "t_m": 60, "t_f": 4e2, "file": "combined_2.csv", "strain_col": "320strainoffset", "time_col": "320timeoffset"},
}

sigma_TS = 515  # Tensile strength (MPa)

# Function to solve for parameters and compute model data
def compute_model(sigma, Ec_0, Ec_m, t_m, t_f):
    sigma_ratio = sigma / sigma_TS

    # Given constants
    c5 = 1.5e-3  # Small constant to avoid log(0)

    # Define A constants
    A1 = np.exp(1)
    A2 = 1 - c5
    A3 = np.exp(-1) - c5
    A4 = -1 / (c5 * np.log(c5))

    # Define system of equations
    def equations(vars):
        c1, c2, c3, c4 = vars
        return [
            Ec_m - (A1 * (sigma_ratio**c3) * np.exp(c4)),
            Ec_0 - (A4 * (sigma_ratio**c3) * np.exp(c4)),
            t_f - (A2 * (sigma_ratio**-c1) * np.exp(-c2)),
            t_m - (A3 * (sigma_ratio**-c1) * np.exp(-c2)),
        ]

    # Solve for c1, c2, c3, c4
    initial_guess = [0.01, -0.1, 0.001, 0.0001]
    solution = root(equations, initial_guess, method='lm')

    if not solution.success:
        raise RuntimeError(f"Solver failed for sigma={sigma}: {solution.message}")

    c1, c2, c3, c4 = solution.x

    # Compute model parameters
    beta = - (sigma_ratio ** c3) * np.exp(c4)
    delta = (sigma_ratio ** c1) * np.exp(c2)
    zeta = c5

    # Time span for simulation
    t_max = max((1 - zeta) / delta, 0.1)
    t_span = (0, min(10000, t_max))
    t_eval = np.linspace(*t_span, 1000)

    # Solve ODE for creep strain
    strain_solution = solve_ivp(creep_rate, t_span, [0], args=(beta, delta, zeta), t_eval=t_eval)

    return strain_solution.t, strain_solution.y[0]

# Solve models dynamically for each sigma in creep_data
model_results = {}
for sigma, values in creep_data.items():
    model_results[sigma] = compute_model(
        sigma, values["Ec_0"], values["Ec_m"], values["t_m"], values["t_f"]
        )

# Load and process experimental data dynamically
exp_data = {}
for sigma, values in creep_data.items():
    data = pd.read_csv(values["file"])
    Time = data[values["time_col"]].to_numpy()
    Strain = data[values["strain_col"]].to_numpy()


    valid_indices = (Time > 0) & (Time <= 203)
    exp_data[sigma] = {
        "Time": Time[valid_indices],
        "Strain": Strain[valid_indices],
    }
    

# ---------------------- Allen's data ---------------------- #  
    
# Replace 'your_data.csv' with the actual file name/path
additional_data = pd.read_csv('combined_2.csv')

# Extract the time and strain columns for the two new curves
christ1 = additional_data['christ1']
chriss1 = additional_data['chriss1']
christ2 = additional_data['christ2']
chriss2 = additional_data['chriss2']

# ---------------------- PLOTTING ALL CURVES---------------------- #
fig, ax1 = plt.subplots()

# Define colors for each sigma value
model_colors = {
    280: '#263675',  # Dark blue
    320: '#00904a',  # Green
    240: '#bc1823',  # Red
    300: '#951abe'   # Purple
}

exp_colors = {
    280: 'lightgrey',  # Light blue
    320: 'lightgrey',  # Light green
    240: 'lightgrey',  # Orange
    300: 'lightgrey'   # Light purple
}


# Plot experimental data dynamically
for sigma, values in exp_data.items():
    ax1.plot(values["Time"], values["Strain"], linestyle='-', color=exp_colors[sigma], label=f'Experimental {sigma} MPa')



ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Creep Strain (%)', fontsize=12)
ax1.tick_params(labelsize=12)


# Create twin y-axis for experimental data
ax2 = ax1.twinx()

# Plot model curves dynamically
for sigma, (time, strain) in model_results.items():
    ax2.plot(time, strain, linestyle='-', color=model_colors[sigma], label=f'Model {sigma} MPa')

ax2.yaxis.set_visible(False)  # Hide y-axis

# Set y-axis limits
ax2.set_ylim(0, 2)
ax1.set_ylim(0, 2)
ax1.set_xlim(-50, 2200)


# Create a joint legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Title and grid

ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
ax1.grid(True, which="major", linestyle=":", linewidth=0.5)

# Turn on minor ticks
ax1.minorticks_on()

# Set automatic minor locators (if needed)
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())

from matplotlib.lines import Line2D

# Create custom legend entry for experimental data
custom_exp_line = Line2D([0], [0], color='lightgrey', lw=2, label='Experimental Data')

# Merge all model and experimental legend handles
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Final legend: one entry for experimental + individual model lines
final_handles = [custom_exp_line] + lines2  # Only show model labels individually
final_labels = [custom_exp_line.get_label()] + labels2

ax1.legend(final_handles, final_labels, loc='upper left', fontsize=9)


plt.savefig("all_curves.png", dpi=300, bbox_inches='tight')
plt.show()


# ---------------------- 280 MPa PLOT ---------------------- #
fig, ax = plt.subplots()



# Plot 280 MPa experimental
ax.plot(exp_data[280]["Time"], exp_data[280]["Strain"],
        linestyle='-', color=exp_colors[280], label='Experimental 280 MPa')

# Plot additional curves
ax.plot(christ1, chriss1, linestyle='-', color='black', label='Additional Curve 1')
ax.plot(christ2, chriss2, linestyle='-', color='gray', label='Additional Curve 2')

# Plot 280 MPa model
ax.plot(model_results[280][0], model_results[280][1],
        linestyle='-', color=model_colors[280], label='Model 280 MPa')



# Labels and styling
ax.set_xlabel(r'Time (hours)', fontsize=12)
ax.set_ylabel(r'Creep Strain (%)', fontsize=12)
ax.set_xlim(-50, 2200)
ax.set_ylim(0, 2)
ax.tick_params(axis='both', labelsize=12)
ax.minorticks_on()
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(True, which="minor", linestyle=":", linewidth=0.5)
ax.grid(True, which="major", linestyle=":", linewidth=0.5)

# Legend
ax.legend(loc='upper left', fontsize=12)

plt.savefig("280_with_allens.png", dpi=300, bbox_inches='tight')
plt.show()



# ====== Function for Zoomed-In Plots ======
def plot_zoomed(sigmas, title):
    fig, ax1 = plt.subplots()
    
    for sigma in sigmas:
        ax1.plot(exp_data[sigma]["Time"], exp_data[sigma]["Strain"], linestyle='-', color=exp_colors[sigma], label=f'Experimental {sigma} MPa')


        
    ax1.set_xlabel(r'Time (hours)', fontsize=12)
    ax1.set_ylabel(r'Creep Strain (%)', fontsize=12)
    ax1.set_xlim(-5, 200)  # Zoom in on time range
    ax1.set_ylim(0, 0.35)  # Keep strain range consistent
    ax1.tick_params(labelsize=12)

    # Twin y-axis for experimental data
    ax2 = ax1.twinx()
    
    # Plot only the selected model curves
    for sigma in sigmas:
        time, strain = model_results[sigma]
        ax2.plot(time, strain, linestyle='-', color=model_colors[sigma], label=f'Model {sigma} MPa')


    # Set up ax2
    ax2.set_yticks([])  
    ax2.yaxis.set_visible(True)
    ax2.set_ylim(0, 0.35)

    # Enable major and minor ticks for ax1
    ax1.minorticks_on()


    # Grid for minor ticks
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
    ax1.grid(True, which="major", linestyle=":", linewidth=0.5)

    # Joint legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=12)
    
    # Save figure
    plt.savefig(f"zoomed_{sigmas[0]}.png", dpi=300, bbox_inches='tight')  # assuming one sigma at a time
    plt.show()

# ====== Zoomed-In Plots ======
plot_zoomed([280], "Zoomed-In: 280 MPa (0-200 hours)")
plot_zoomed([320], "Zoomed-In: 320 MPa (0-200 hours)")
plot_zoomed([240], "Zoomed-In: 240 MPa (0-200 hours)")
plot_zoomed([300], "Zoomed-In: 300 MPa (0-200 hours)")

# ---------------------- ADDITIONAL COMPARISON PLOTS ---------------------- #
sigma_ratios = [s / sigma_TS for s in creep_data.keys()]
ec_0_values = [values["Ec_0"] for values in creep_data.values()]
ec_m_values = [values["Ec_m"] for values in creep_data.values()]
t_m_values = [values["t_m"] for values in creep_data.values()]
t_f_values = [values["t_f"] for values in creep_data.values()]

# Define function to plot scatter plots with logarithmic axes
def power_law(x, a, b):
    return a * np.power(x, b)

def plot_scatter_log(x, y1, y2, title, ylabel, marker1, label1, marker2, label2, filename):
    fig, ax = plt.subplots()
    
    # Scatter points
    ax.scatter(x, y1, marker=marker1, color='#bc1823', label=label1)
    ax.scatter(x, y2, marker=marker2, color='#263675', label=label2)

    # Fit and plot trendline for y1
    popt1, _ = curve_fit(power_law, x, y1)
    x_fit = np.linspace(min(x), max(x), 200)
    y1_fit = power_law(x_fit, *popt1)
    ax.plot(x_fit, y1_fit, linestyle='--', linewidth=1,color='#bc1823', label=f'{label1} Trend')

    # Fit and plot trendline for y2
    popt2, _ = curve_fit(power_law, x, y2)
    y2_fit = power_law(x_fit, *popt2)
    ax.plot(x_fit, y2_fit, linestyle='--', linewidth=1 ,color='#263675', label=f'{label2} Trend')

    # Labels and formatting
    ax.set_xlabel(r'Stress Ratio', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# === Plot 1: ε₀ᶜ and εₘᶜ ===
plot_scatter_log(
    sigma_ratios,
    ec_0_values, ec_m_values,
    r'$\epsilon_0^{c}$ & $\epsilon_m^{c}$ vs Stress Ratio', 
    'Creep Strain (%)',
    'o', r'$\epsilon_0^{c}$',
    's', r'$\epsilon_m^{c}$',
    'ec_e0_curves.png'
)

# === Plot 2: t_m and t_f ===
plot_scatter_log(
    sigma_ratios,
    t_m_values, t_f_values,
    r'$t_m$ & $t_f$ vs Stress Ratio', 
    'Creep Time (hours)',
    'o', r'$t_m$',
    's', r'$t_f$',
    'tm_tf_curves.png'
)


# ---------------------- error and R squared ---------------------- #

# def compute_errors_and_r2(exp_time, exp_strain, model_time, model_strain):
#     # Ensure model time covers experimental time range
#     valid_model_indices = (model_time >= exp_time[0]) & (model_time <= exp_time[-1])
#     model_time = model_time[valid_model_indices]
#     model_strain = model_strain[valid_model_indices]
    
#     # Find the closest model time points to experimental time points
#     model_indices = np.searchsorted(model_time, exp_time, side='left')
#     model_indices = np.clip(model_indices, 0, len(model_time) - 1)
#     model_strain_cropped = model_strain[model_indices]
    
#     # Avoid division by zero or extremely small experimental strain values
#     exp_strain = np.where(exp_strain < 1e-6, 1e-6, exp_strain)  # Prevents huge errors
#     valid_indices = exp_strain[1:] != 0  # Ensure no division by zero
    
#     # Compute errors safely
#     error = np.where(valid_indices, ((exp_strain[1:] - model_strain_cropped[1:]) / exp_strain[1:]) * 100, np.nan)
#     abs_error = np.abs(error)
    
#     # Compute mean errors while ignoring NaN values
#     mean_error = np.nanmean(error)
#     mean_abs_error = np.nanmean(abs_error)

#     # Compute R-squared value safely
#     ss_total = np.nansum((exp_strain - np.nanmean(exp_strain)) ** 2)
#     ss_residual = np.nansum((exp_strain - model_strain_cropped) ** 2)
    
#     if ss_total > 0:
#         r_squared = 1 - (ss_residual / ss_total)
#     else:
#         r_squared = 0  # If no variance, set R² to 0 instead of NaN

#     return mean_error, mean_abs_error, r_squared

# # Calculate errors and R-squared for each stress level
# error_results = {}
# for sigma in creep_data.keys():
#     exp_time = exp_data[sigma]['Time']
#     exp_strain = exp_data[sigma]['Strain']
#     model_time, model_strain = model_results[sigma]
    
#     mean_error, mean_abs_error, r_squared = compute_errors_and_r2(exp_time, exp_strain, model_time, model_strain)
    
#     error_results[sigma] = {
#         'Mean Error (%)': mean_error, 
#         'Mean Absolute Error (%)': mean_abs_error,
#         'R-squared': r_squared
#     }
    
#     print(f"{sigma} MPa:")
#     print(f"Mean Error (%): {mean_error:.4f}")
#     print(f"Mean Absolute Error (%): {mean_abs_error:.4f}")
#     print(f"R-squared: {r_squared:.4f}")




