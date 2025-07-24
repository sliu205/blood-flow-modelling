import re
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===== USER CONFIGURATION =====
INPUT_FILE = "/home/sliu205/Downloads/HPC ouputs/Kapela_Vm_calcium_trace/Kapela_11_6_25_redone_cost_0.233/0.233_cost_regeneration"
OUTPUT_DIR = "./sensitivity_results"  # Where to save plots and summary

# Analysis thresholds
HIGH_THRESHOLD = 7               # Upper bound for log plots
SENSITIVITY_THRESHOLD = 0.2      # Cost deviation threshold for classification
LOG_LOW_THRESHOLD = 0.5          # Lower bound for near-insensitive parameters
PLOT_COST_THRESHOLD = 0.2        # Visual threshold for plots (± fraction)
# ==============================

def parse_sensitivity_output(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    base_cost_match = re.search(r'Total Gaussian[-\‐]MLE cost = (\d+\.\d+)', content)
    if base_cost_match:
        base_cost = float(base_cost_match.group(1))
    else:
        raise ValueError("Could not find base cost in the output file.")

    param_sections = re.findall(r"=== Sweeping '(.*?)'.*?best_param = (.*?) ===(.*?)(?===|$)", content, re.DOTALL)

    data = []
    for param_name, best_param, section in param_sections:
        best_param = float(best_param)
        variations = re.findall(rf'\s+{re.escape(param_name)} = (.*?) \((.*?)x\)\s+→\s+cost = (.*?)\n', section)

        if not variations:
            continue

        param_data = {'Parameter': param_name, 'best_param': best_param}
        for val, x, cost in variations:
            param_data[f'{x}x_param'] = float(val)
            param_data[f'{x}x_cost'] = float(cost)
        data.append(param_data)

    return pd.DataFrame(data), base_cost

def analyze_sensitivity(df, base_cost):
    fully, partial, insensitive = [], [], []
    
    for _, row in df.iterrows():
        param = row['Parameter']
        cost_cols = [col for col in row.index if col.endswith('x_cost')]
        deviations = [abs(row[col] - base_cost) > SENSITIVITY_THRESHOLD * base_cost for col in cost_cols]

        if all(deviations):
            fully.append(param)
        elif any(deviations):
            partial.append(param)
        else:
            insensitive.append(param)

    return fully, partial, insensitive

def plot_results(df, base_cost, params=None, scale='linear', high=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plot_df = df[df['Parameter'].isin(params)] if params else df
    melted = plot_df.melt(id_vars=['Parameter'],
                         value_vars=[col for col in plot_df.columns if 'x_cost' in col],
                         var_name='Variation', value_name='Cost')
    melted['x_value'] = melted['Variation'].str.extract(r'([\d\.]+)x_cost').astype(float)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for param in plot_df['Parameter']:
        pdata = melted[melted['Parameter'] == param]
        ax.plot(pdata['x_value'], pdata['Cost'], 'o-', label=param)

    ax.axhline(base_cost, color='k', linestyle='--', label='Baseline')
    ax.axhline(base_cost*(1-PLOT_COST_THRESHOLD), color='r', linestyle=':', 
               label=f'±{PLOT_COST_THRESHOLD*100:.0f}% Threshold')
    ax.axhline(base_cost*(1+PLOT_COST_THRESHOLD), color='r', linestyle=':')
    
    ax.set_yscale(scale)
    if scale == 'log' and high:
        ax.set_ylim(top=base_cost*high*1.1)
    
    ax.set_title(f"Sensitivity Analysis ({scale.capitalize()} Scale)")
    ax.set_xlabel('Parameter Variation (x)')
    ax.set_ylabel('Cost')
    ax.grid(True)
    
    # Place legend outside the plot to the right
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             borderaxespad=0., fontsize='small')
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"sensitivity_{scale}_{'filtered' if params else 'all'}.pdf"
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def get_log_plot_params(df, base_cost):
    near_insensitive = []
    for _, row in df.iterrows():
        param = row['Parameter']
        cost_cols = [col for col in row.index if col.endswith('x_cost')]
        costs = [row[col] for col in cost_cols]
        if all(LOG_LOW_THRESHOLD*base_cost <= cost <= HIGH_THRESHOLD*base_cost for cost in costs):
            near_insensitive.append(param)
    return near_insensitive

def run_parameter_modifications_loop(df, base_cost):
    """
    Loop to ask user for parameter changes repeatedly, analyze, and plot results after each.
    """
    previous_df = df.copy()
    previous_base_cost = base_cost
    modification_number = 1

    while True:
        print(f"\n=== Modification {modification_number} ===")
        param_to_modify = input("Enter parameter name to modify (or press Enter to finish): ").strip()
        if param_to_modify == "":
            print("Finished all parameter modifications.")
            break

        if param_to_modify not in df['Parameter'].values:
            print(f"⚠️ Parameter '{param_to_modify}' not found in dataset. Try again.")
            continue

        factor_str = input(f"Enter multiplicative factor for '{param_to_modify}' (e.g. 1.2): ").strip()
        try:
            factor = float(factor_str)
        except ValueError:
            print("⚠️ Invalid factor input. Please enter a numeric value.")
            continue

        # Apply modification: multiply best_param of param_to_modify by factor
        mod_df = previous_df.copy()
        mod_df.loc[mod_df['Parameter'] == param_to_modify, 'best_param'] *= factor

        # *** Note: You can insert real recalculation or reload logic here if needed ***
        # For now, we just simulate recalculation by adjusting 'best_param' (as placeholder)

        # You could recalc costs or load new df here if your workflow supports it.
        # For now, let's keep previous costs and just print info.

        print(f"Applied factor {factor} to parameter '{param_to_modify}'.")
        # Show updated param best value
        print(f"New best_param for '{param_to_modify}': {mod_df.loc[mod_df['Parameter'] == param_to_modify, 'best_param'].values[0]}")

        # Analyze sensitivity on updated dataframe - you may want to update this with new data if available
        fully, partial, insensitive = analyze_sensitivity(mod_df, previous_base_cost)

        print(f"\nSensitivity classification after modification {modification_number}:")
        print(f"Fully sensitive ({len(fully)}): {fully}")
        print(f"Partially sensitive ({len(partial)}): {partial}")
        print(f"Insensitive ({len(insensitive)}): {insensitive}")

        # Plotting updated results (passing mod_df)
        plot_results(mod_df, previous_base_cost, scale='linear')
        plot_results(mod_df, previous_base_cost, scale='log', high=HIGH_THRESHOLD)

        log_params = get_log_plot_params(mod_df, previous_base_cost)
        if log_params:
            plot_results(mod_df, previous_base_cost, params=log_params, scale='log', high=HIGH_THRESHOLD)
            print(f"\nParameters for filtered log plot: {log_params}")

        modification_number += 1
        previous_df = mod_df.copy()

def main():
    print(f"Analyzing: {INPUT_FILE}")
    df, base_cost = parse_sensitivity_output(INPUT_FILE)
    
    fully, partial, insensitive = analyze_sensitivity(df, base_cost)
    print(f"\nFully sensitive ({len(fully)}): {fully}")
    print(f"Partially sensitive ({len(partial)}): {partial}")
    print(f"Insensitive ({len(insensitive)}): {insensitive}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    summary_path = os.path.join(OUTPUT_DIR, "sensitivity_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Base Cost: {base_cost}\n\n")
        f.write(f"Fully sensitive ({len(fully)}):\n" + "\n".join(fully) + "\n\n")
        f.write(f"Partially sensitive ({len(partial)}):\n" + "\n".join(partial) + "\n\n")
        f.write(f"Insensitive ({len(insensitive)}):\n" + "\n".join(insensitive))
    print(f"\nSummary saved to: {summary_path}")
    
    print("\nGenerating initial plots...")
    plot_results(df, base_cost, scale='linear')
    plot_results(df, base_cost, scale='log', high=HIGH_THRESHOLD)
    
    log_params = get_log_plot_params(df, base_cost)
    print(f"\nParameters for filtered log plot: {log_params}")
    plot_results(df, base_cost, params=log_params, scale='log', high=HIGH_THRESHOLD)
    
    print(f"\nAll initial results saved to: {os.path.abspath(OUTPUT_DIR)}")

    # Start interactive parameter modification loop
    run_parameter_modifications_loop(df, base_cost)

if __name__ == "__main__":
    main()

