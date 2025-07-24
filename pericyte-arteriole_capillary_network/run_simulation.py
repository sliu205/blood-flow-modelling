#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from opencor_helper import SimulationHelper
import os
from datetime import datetime

# ===== USER-CONFIGURABLE PARAMETERS =====
CELLML_FILE = "/home/sliu205/Documents/git_projects/pericyte-arteriole_capillary_network/generated_models/microvasculature_network/microvasculature_network.cellml"
SIM_DT = 0.01          # Simulation time step
SIM_TIME = 2.5         # Total simulation time
PRE_TIME = 0           # Pre-simulation time
MAX_TIMESTEP = 0.01    # Maximum solver time step (critical for stability)
OUTPUT_DIR = "simulation_results"

# Percent change plot settings
PERCENT_Y_MIN = -100     # Minimum y-axis value (%)
PERCENT_Y_MAX = 50      # Maximum y-axis value (%)
PERCENT_Y_INCREMENT = 20  # Y-axis increment (%)
# ========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_unit(simulation, varname):
    """Get the unit for a variable from the simulation object"""
    try:
        if varname == 'time':
            return simulation.data().voi().unit()
        elif varname in simulation.results().states():
            return simulation.results().states()[varname].unit()
        elif varname in simulation.results().algebraic():
            return simulation.results().algebraic()[varname].unit()
        elif varname in simulation.data().constants():
            constant = simulation.data().constants()[varname]
            if hasattr(constant, 'unit'):
                return constant.unit()
            return ""
        return ""
    except Exception as e:
        print(f"Warning: Could not get unit for {varname}: {str(e)}")
        return ""

def format_unit(unit_str):
    """Format the unit string for display"""
    if not unit_str:
        return ""
    return f" ({unit_str})"

def extract_var_name(full_name):
    """Extract just the variable name after the last '/'"""
    return full_name.split('/')[-1]

def discover_and_select_variables():
    #print("Discovering available variables...")
    sim = SimulationHelper(
        CELLML_FILE, SIM_DT, 10,
        solver_info={'MaximumStep': MAX_TIMESTEP, 'MaximumNumberOfSteps': 500000},
        pre_time=0
    )
    state_vars = list(sim.simulation.results().states())
    algebraic_vars = list(sim.simulation.results().algebraic())
    constant_vars = list(sim.data.constants())
    
    # Get units for each variable
    units_info = {}
    for v in state_vars + algebraic_vars + constant_vars + ['time']:
        units_info[v] = get_unit(sim.simulation, v)
    
    all_vars = state_vars + algebraic_vars + constant_vars
    sim.close_simulation()

   # print("\n=== AVAILABLE VARIABLES ===")
    #for i, v in enumerate(all_vars):
        #print(f"{i:3}: {v}{format_unit(units_info.get(v, ''))}")

    input_str = input("\nEnter comma-separated variable names to PLOT (e.g. TC1/u, TC2/u):\n> ")
    requested_vars = [v.strip() for v in input_str.split(',')]
    valid_vars = ['time'] + [v for v in requested_vars if v in all_vars]
    invalid_vars = [v for v in requested_vars if v not in all_vars]

    if invalid_vars:
        print(f"\n⚠️ Invalid variable(s): {', '.join(invalid_vars)}")
    if len(valid_vars) <= 1:
        print("❌ No valid variables selected. Exiting.")
        exit()

    return valid_vars, all_vars, units_info

def prompt_params_to_modify(all_vars, units_info):
    print("\n=== Select parameter(s) to MODIFY (states or constants) or press Enter to skip ===")
    #print("Available parameters:")
    #for i, v in enumerate(all_vars):
        #print(f"{i:3}: {v}{format_unit(units_info.get(v, ''))}")

    input_str = input("\nEnter comma-separated parameter names to MODIFY (or empty to skip):\n> ").strip()
    if input_str == "":
        return [], [], [], []

    param_names = [v.strip() for v in input_str.split(',')]

    invalid = [p for p in param_names if p not in all_vars]
    if invalid:
        print(f"⚠️ Invalid parameter(s): {', '.join(invalid)}")
        exit()

    mod_factors = []
    mod_types = []
    mod_labels = []

    print("\nFor each parameter, enter either:")
    print(" - a scaling factor (e.g. 1.2 to multiply by 1.2)")
    print(" - or an absolute value prefixed by '=' (e.g. '=0.5') to set directly")
    for p in param_names:
        val_str = input(f"Modify '{p}': ")
        if val_str.startswith('='):
            try:
                val = float(val_str[1:])
                mod_factors.append(val)
                mod_types.append('absolute')
                mod_labels.append(f"{p}={val}{format_unit(units_info.get(p, ''))}")
            except:
                print("Invalid input, exiting")
                exit()
        else:
            try:
                val = float(val_str)
                mod_factors.append(val)
                mod_types.append('scaling')
                mod_labels.append(f"{p}×{val}")
            except:
                print("Invalid input, exiting")
                exit()

    return param_names, mod_factors, mod_types, mod_labels

def save_steady_state_bar_plot(selected_vars, results, time, timestamp, units_info, suffix=""):
    variable_names = selected_vars[1:]
    start_idx = int(0.75 * len(time))
    steady_values = [np.mean(results[i][0][start_idx:]) for i in range(1, len(selected_vars))]

    plt.figure(figsize=(10, 5))
    plt.bar(variable_names, steady_values, color='teal')
    plt.xticks(rotation=45, ha='right')
    
    # Use the base variable name (after last '/') and unit for y-axis
    base_var_name = extract_var_name(selected_vars[1])
    y_label = f"{base_var_name}{format_unit(units_info.get(selected_vars[1], ''))}"
    plt.ylabel(y_label)
    
    plt.title("Steady-State Estimates (Final 25% of Simulation)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"bar_plot_{timestamp}{suffix}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Bar plot saved to {path}")

def run_and_compare_modifications(selected_vars, units_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"session_{timestamp}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Initialize simulation for baseline
    sim = SimulationHelper(
        CELLML_FILE, SIM_DT, SIM_TIME,
        solver_info={'MaximumStep': MAX_TIMESTEP, 'MaximumNumberOfSteps': 500000},
        pre_time=PRE_TIME
    )
    
    # Store baseline parameters
    baseline_param_vals = {}
    all_mod_results = []  # Store (label, results) for each modification
    mod_labels = []
    
    # Run baseline simulation
    print("\nRunning baseline simulation...")
    if not sim.run():
        print("Baseline simulation failed!")
        sim.close_simulation()
        return
    
    baseline_results = sim.get_results(selected_vars)
    time = baseline_results[0][0]
    
    # Save baseline results
    with open(os.path.join(session_dir, f"baseline_results.csv"), 'w') as f:
        header = ["Time" + format_unit(units_info.get('time', ''))]
        header += [var + format_unit(units_info.get(var, '')) for var in selected_vars[1:]]
        f.write(",".join(header) + "\n")
        for i in range(len(time)):
            row = [str(time[i])] + [str(baseline_results[j][0][i]) for j in range(1, len(selected_vars))]
            f.write(",".join(row) + "\n")
    print(f"✅ Baseline data saved to session_{timestamp}/baseline_results.csv")
    
    # Plot baseline
    plt.figure(figsize=(12, 6))
    for i, var in enumerate(selected_vars[1:], 1):
        plt.plot(time, baseline_results[i][0], label=var)
    
    time_unit = format_unit(units_info.get('time', ''))
    plt.xlabel(f"Time{time_unit}")
    
    base_var_name = extract_var_name(selected_vars[1])
    y_label = f"{base_var_name}{format_unit(units_info.get(selected_vars[1], ''))}"
    plt.ylabel(y_label)
    
    plt.title("Baseline Simulation Results")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, f"baseline_plot.png"), dpi=300)
    plt.close()
    
    # Main loop for modifications
    modification_count = 0
    while True:
        print("\n" + "="*50)
        print(f"MODIFICATION {modification_count + 1}")
        print("="*50)
        
        # Prompt for modification parameters
        param_names, mod_factors, mod_types, current_mod_labels = prompt_params_to_modify(
            list(sim.data.constants().keys()) + list(sim.simulation.results().states()), 
            units_info
        )
        
        if not param_names:
            print("\nNo parameters selected for modification. Exiting modification loop.")
            break
            
        # Reset to baseline
        sim.reset_and_clear()
        
        # Apply modification
        absolute_flags = [t == 'absolute' for t in mod_types]
        if any(absolute_flags):
            # Handle mixed absolute and relative modifications
            new_vals = []
            for i, p in enumerate(param_names):
                if mod_types[i] == 'absolute':
                    new_vals.append(mod_factors[i])
                else:
                    # For relative changes, get baseline value
                    base_val = sim.get_init_param_vals([p])[0]
                    new_vals.append(base_val * mod_factors[i])
            sim.set_param_vals(param_names, new_vals)
        else:
            # All relative changes
            base_vals = sim.get_init_param_vals(param_names)
            new_vals = [base_vals[i] * mod_factors[i] for i in range(len(param_names))]
            sim.set_param_vals(param_names, new_vals)
        
        # Run modified simulation
        mod_label = ", ".join(current_mod_labels)
        print(f"\nRunning simulation with modification: {mod_label}")
        if not sim.run():
            print("Modified simulation failed!")
            continue
            
        mod_results = sim.get_results(selected_vars)
        all_mod_results.append((mod_label, mod_results))
        mod_labels.append(mod_label)
        modification_count += 1
        
        # Save modified results
        mod_filename = f"mod_{modification_count}_results.csv"
        with open(os.path.join(session_dir, mod_filename), 'w') as f:
            header = ["Time" + format_unit(units_info.get('time', ''))]
            header += [var + format_unit(units_info.get(var, '')) for var in selected_vars[1:]]
            f.write(",".join(header) + "\n")
            for i in range(len(time)):
                row = [str(time[i])] + [str(mod_results[j][0][i]) for j in range(1, len(selected_vars))]
                f.write(",".join(row) + "\n")
        print(f"✅ Modified data saved to {mod_filename}")
        
        # Ask if user wants to perform another modification
        another = input("\nPerform another modification? (y/n): ").strip().lower()
        if another != 'y':
            break
    
    sim.close_simulation()
    
    # Generate comparison plots if we have modifications
    if modification_count > 0:
        # Time series comparison plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_vars[1:])))
        
        # Plot baseline
        for i, var in enumerate(selected_vars[1:], 1):
            plt.plot(time, baseline_results[i][0], label=f'Baseline: {var}', 
                     color=colors[i-1], linestyle='-', linewidth=2)
        
        # Plot modifications
        for mod_idx, (label, results) in enumerate(all_mod_results):
            for i, var in enumerate(selected_vars[1:], 1):
                plt.plot(time, results[i][0], label=f'Mod{mod_idx+1}: {var} ({label})', 
                         color=colors[i-1], linestyle=['--', '-.', ':'][mod_idx % 3], 
                         alpha=0.8)
        
        plt.xlabel(f"Time{format_unit(units_info.get('time', ''))}")
        base_var_name = extract_var_name(selected_vars[1])
        y_label = f"{base_var_name}{format_unit(units_info.get(selected_vars[1], ''))}"
        plt.ylabel(y_label)
        plt.title("Comparison of All Modifications")
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, f"all_modifications_comparison.png"), dpi=300)
        plt.close()
        print(f"✅ All modifications comparison plot saved")
        
        # Steady-state comparison (bar plot)
        start_idx = int(0.75 * len(time))
        num_vars = len(selected_vars[1:])
        num_mods = len(all_mod_results)
        
        plt.figure(figsize=(12, 8))
        bar_width = 0.8 / (num_mods + 1)
        x = np.arange(num_vars)
        
        # Baseline bars
        baseline_steady = [np.mean(baseline_results[i][0][start_idx:]) for i in range(1, len(selected_vars))]
        plt.bar(x, baseline_steady, width=bar_width, color='blue', label='Baseline')
        
        # Modification bars
        for mod_idx, (label, results) in enumerate(all_mod_results):
            mod_steady = [np.mean(results[i][0][start_idx:]) for i in range(1, len(selected_vars))]
            offset = bar_width * (mod_idx + 1)
            plt.bar(x + offset, mod_steady, width=bar_width, 
                    label=f'Mod{mod_idx+1}: {label}')
        
        # Use FULL variable names for x-axis labels (not shortened)
        plt.xticks(x + bar_width * (num_mods + 1) / 2, 
                   selected_vars[1:], 
                   rotation=45, ha='right')
        
        base_var_name = extract_var_name(selected_vars[1])
        y_label = f"{base_var_name}{format_unit(units_info.get(selected_vars[1], ''))}"
        plt.ylabel(y_label)
        plt.title("Steady-State Comparison (Final 25% of Simulation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, f"steady_state_comparison.png"), dpi=300)
        plt.close()
        print(f"✅ Steady-state comparison plot saved")
        
        # Percent change plot
        plt.figure(figsize=(12, 8))
        
        for mod_idx, (label, results) in enumerate(all_mod_results):
            percent_changes = []
            for var_idx in range(1, len(selected_vars)):
                base_val = np.mean(baseline_results[var_idx][0][start_idx:])
                mod_val = np.mean(results[var_idx][0][start_idx:])
                percent = 100 * (mod_val - base_val) / base_val if base_val != 0 else 0
                percent_changes.append(percent)
            
            plt.bar(np.arange(num_vars) + mod_idx * 0.2, percent_changes, 
                    width=0.2, label=f'Mod{mod_idx+1}: {label}')
        
        plt.axhline(0, color='black', linewidth=0.8)
        
        # Use FULL variable names for x-axis labels (not shortened)
        plt.xticks(np.arange(num_vars) + 0.2 * (num_mods - 1) / 2, 
                   selected_vars[1:], 
                   rotation=45, ha='right')
        
        plt.ylabel("% Change from Baseline")
        plt.ylim(PERCENT_Y_MIN, PERCENT_Y_MAX)
        plt.yticks(np.arange(PERCENT_Y_MIN, PERCENT_Y_MAX + PERCENT_Y_INCREMENT, PERCENT_Y_INCREMENT))
        plt.title("Percent Change from Baseline")
        plt.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(session_dir, f"percent_change_comparison.png"), dpi=300)
        plt.close()
        print(f"✅ Percent change comparison plot saved")
        
        print(f"\nAll results saved in directory: {session_dir}")

def main():
    selected_vars, all_vars, units_info = discover_and_select_variables()
    run_and_compare_modifications(selected_vars, units_info)

if __name__ == "__main__":
    main()

