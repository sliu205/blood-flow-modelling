#!/usr/bin/env python3

import sys
import os
import yaml
import pandas as pd
from mpi4py import MPI

# Add src directory to path
root_dir = os.path.join(os.path.dirname(__file__), "../..")
sys.path.append(os.path.join(root_dir, "src"))

from param_id.paramID import CVS0DParamID
from parsers.PrimitiveParsers import YamlFileParser

def load_yaml_to_dict(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    yaml_file = '/people/sliu205/Kapela_calibration/Kapela_CA/Kapela_JUNE_25_user_inputs.yaml'
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"YAML file not found: {yaml_file}")

    # Load YAML file into dictionary before parsing
    inp_dict = load_yaml_to_dict(yaml_file)

    # Parse inputs (pass dict, not filename)
    parser = YamlFileParser()
    inp = parser.parse_user_inputs_file(
        inp_dict,
        obs_path_needed=True,
        do_generation_with_fit_parameters=True
    )

    # Fix missing model_path if needed
    if 'model_path' not in inp or not inp['model_path']:
        inp['model_path'] = os.path.join(inp['resources_dir'], 'Kapela.cellml')

    # Initialize parameter identification class
    param_id = CVS0DParamID(
        model_path=inp['model_path'],
        model_type=inp['model_type'],
        param_id_method=inp['param_id_method'],
        mcmc_instead=False,
        file_name_prefix=inp['file_prefix'],
        params_for_id_path=inp['params_for_id_path'],
        param_id_obs_path=inp['param_id_obs_path'],
        sim_time=inp['sim_time'],
        pre_time=inp['pre_time'],
        solver_info=inp['solver_info'],
        ga_options=inp.get('ga_options'),
        dt=inp['dt'],
        param_id_output_dir=inp['param_id_output_dir'],
        resources_dir=inp['resources_dir']
    )

    # Run simulation with best param values
    param_id.simulate_with_best_param_vals()

    # Extract outputs and merge into dataframe
    cost, obs_outputs = param_id.param_id.get_cost_and_obs_from_params(param_id.param_id.best_param_vals)

    # Get time vectors per subexperiment
    if hasattr(param_id.param_id.sim_helper, "get_time_vectors"):
        time_vectors = param_id.param_id.sim_helper.get_time_vectors()
    else:
        # Fallback: use protocol_info time vectors if available
        if hasattr(param_id.param_id, "protocol_info") and "time" in param_id.param_id.protocol_info:
            time_vectors = param_id.param_id.protocol_info["time"]
        else:
            raise AttributeError("No method 'get_time_vectors' and no protocol_info time vector found.")

    obs_names = param_id.obs_info["obs_names"]

    # Create a list of dataframes for each observable and subexperiment
    records = []
    for sub_idx, series in enumerate(obs_outputs):
        series_data = param_id.param_id.get_obs_output_dict(series)['series']
        for obs_i, data in enumerate(series_data):
            time = time_vectors[sub_idx]
            label = f'exp{sub_idx}_{obs_names[obs_i]}'
            records.append(pd.DataFrame({'time': time, label: data}))

    # Merge all dataframes on 'time'
    df = records[0]
    for r in records[1:]:
        df = pd.merge(df, r, on='time', how='outer')

    # Merge ground truth series if available
    gt_df = getattr(param_id, "gt_df", None)
    if gt_df is not None:
        for _, row in gt_df.iterrows():
            if row['data_type'] == 'series':
                name = row['variable']
                label = f'exp{row.get("subexperiment_idx", 0)}_{name}_gt'
                series = row.get('ground_truth_series') or row.get('series')
                if series is not None:
                    time_len = len(series)
                    subexp_idx = int(row.get('subexperiment_idx', 0))
                    time = time_vectors[subexp_idx][:time_len]
                    df_gt = pd.DataFrame({'time': time, label: series})
                    df = pd.merge(df, df_gt, on='time', how='outer')

    # Save combined data to CSV file
    output_csv = os.path.join(param_id.output_dir, f'{inp["file_prefix"]}_sim_and_gt.csv')
    df.to_csv(output_csv, index=False)
    print(f'Data saved to {output_csv}')

    # Close simulation resources
    param_id.close_simulation()


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        import traceback
        print(traceback.format_exc())
        comm.Abort()
        exit()
