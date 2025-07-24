#!/usr/bin/env python3

import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opencor_helper import SimulationHelper

# === USER CONFIGURABLE SETTINGS ===
CELLML_FILE = "/home/sliu205/Documents/git_projects/pericyte-arteriole_capillary_network/generated_models/microvasculature_network/microvasculature_network.cellml"

# Time settings
PHASE_DURATION = 10.0  # seconds per phase
SIM_DT = 0.01
MAX_TIMESTEP = 0.01
OUTPUT_DIR = "multi_phase_results"

# Parameter settings
PARAM_TO_MODIFY = "PTC9/r_0"
CONSTRICTION_SCALING = 0.1
DILATION_SCALING = 2.0

# Variables to plot
PLOT_VARS = ["time", "TC9/u", "B1V1/u", "B2V1/u"]
# ===================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize simulation helper
sim = SimulationHelper(
    CELLML_FILE,
    dt=SIM_DT,
    sim_time=PHASE_DURATION,
    solver_info={'MaximumStep': MAX_TIMESTEP, 'MaximumNumberOfSteps': 500000},
    pre_time=0.0
)

sim.reset_and_clear()

# Save original parameter value
original_value = sim.get_init_param_vals([PARAM_TO_MODIFY])[0]

# === PHASE FUNCTION ===
def run_phase(sim, start_time, sim_time, label, param_value=None):
    if param_value is not None:
        sim.set_param_vals([PARAM_TO_MODIFY], [param_value])
    sim.update_times(dt=SIM_DT, start_time=start_time, sim_time=sim_time, pre_time=0.0)
    if not sim.run():
        print(f"❌ {label} phase failed.")
        sim.close_simulation()
        exit()
    results = sim.get_results(PLOT_VARS)
    time = results[0][0]
    data = [results[i][0] for i in range(1, len(PLOT_VARS))]
    return time, data

# === PHASES ===
times, datas = [], []
current_time = 0.0

# Store phase names in order, matching the sequence of phases and vertical lines
phase_names = [
    "Rest (1)",
    "Constriction",
    "Rest (2)",
    "Dilation",
    "Rest (3)"
]

# 1. Rest (original parameter)
t, d = run_phase(sim, current_time, PHASE_DURATION, phase_names[0], param_value=original_value)
times.append(t)
datas.append(d)
current_time += PHASE_DURATION

# 2. Constriction
constricted_value = original_value * CONSTRICTION_SCALING
t, d = run_phase(sim, current_time, PHASE_DURATION, phase_names[1], param_value=constricted_value)
times.append(t)
datas.append(d)
current_time += PHASE_DURATION

# 3. Rest (return to baseline)
t, d = run_phase(sim, current_time, PHASE_DURATION, phase_names[2], param_value=original_value)
times.append(t)
datas.append(d)
current_time += PHASE_DURATION

# 4. Dilation
dilated_value = original_value * DILATION_SCALING
t, d = run_phase(sim, current_time, PHASE_DURATION, phase_names[3], param_value=dilated_value)
times.append(t)
datas.append(d)
current_time += PHASE_DURATION

# 5. Rest (return to baseline)
t, d = run_phase(sim, current_time, PHASE_DURATION, phase_names[4], param_value=original_value)
times.append(t)
datas.append(d)
current_time += PHASE_DURATION

# === Combine results ===
full_time = np.concatenate(times)
full_data = [np.concatenate([d[i] for d in datas]) for i in range(len(PLOT_VARS) - 1)]

# === Extract unit dynamically for y-axis label ===

def get_unit_from_cellml(var_name):
    try:
        algebraic_vars = sim.simulation.results().algebraic()
        if var_name in algebraic_vars:
            unit = algebraic_vars[var_name].unit()
            if unit:
                return unit
        states_vars = sim.simulation.results().states()
        if var_name in states_vars:
            unit = states_vars[var_name].unit()
            if unit:
                return unit
    except Exception as e:
        print(f"Warning: Could not retrieve unit for {var_name} from CellML model: {e}")
    return None

def parse_unit_suffix(var_name):
    if "/" in var_name:
        return var_name.split("/")[-1]
    return ""

first_var = PLOT_VARS[1]
unit_from_model = get_unit_from_cellml(first_var)
unit_suffix = parse_unit_suffix(first_var)
unit_label = unit_from_model if unit_from_model else unit_suffix

# === Plotting ===
plt.figure(figsize=(10, 6))
for i, var in enumerate(PLOT_VARS[1:]):
    plt.plot(full_time, full_data[i], label=var)
    
# Add vertical lines and phase text at phase transitions including the start
for i in range(len(phase_names)):
    xpos = i * PHASE_DURATION
    if i > 0:
        plt.axvline(xpos, color='red', linestyle='--')
    ylim = plt.ylim()
    y_text = ylim[0] + 0.03 * (ylim[1] - ylim[0])  # 3% above bottom

    # For the first label, place text slightly right of xpos=0
    # For others, place text slightly right of vertical lines
    if i == 0:
        plt.text(xpos + 0.2, y_text, phase_names[i], color='red', fontsize=9, ha='left', va='bottom')
    else:
        plt.text(xpos + 0.2, y_text, phase_names[i], color='red', fontsize=9, ha='left', va='bottom')


plt.xlabel("Time (s)", labelpad=15)  # add extra padding below xlabel

# Move xlabel downward slightly to separate from phase text
ax = plt.gca()
# Current label coords (0.5, 0) center bottom, shift down a bit
ax.xaxis.set_label_coords(0.5, -0.07)  # 7% below x-axis line

plt.ylabel(f"{first_var.split('/')[-1]} ({unit_label})")
plt.title(f"Multi-phase simulation\n({PARAM_TO_MODIFY}, Constriction ×{CONSTRICTION_SCALING}, Dilation ×{DILATION_SCALING})")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, f"time_trace_{timestamp}.png")
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"✅ Plot saved: {plot_path}")



# === CSV Export ===
csv_path = os.path.join(OUTPUT_DIR, f"combined_results_{timestamp}.csv")
with open(csv_path, 'w') as f:
    header = "Time(s)," + ",".join(PLOT_VARS[1:]) + "\n"
    f.write(header)
    for i in range(len(full_time)):
        row = [str(full_time[i])] + [str(full_data[j][i]) for j in range(len(full_data))]
        f.write(",".join(row) + "\n")
print(f"✅ Data saved: {csv_path}")

sim.close_simulation()

