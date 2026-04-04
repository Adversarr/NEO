"""
In the csv file:
experiment_id,experiment_name,run_id,run_name,metric_key,step,value,timestamp
911542225448014930,cls_shrec16,10568c4c1cee452dba52e937ca0087bd,with-pretrain-240,Val/Acc,299,0.06666667014360428,1768479359125
...

Key differences from previous format:
- 'run_name' instead of 'Run'
- 'metric_key' instead of 'metric'
- 'run_id' instead of 'Run ID'
- The rest of the logic should be similar.

90, 30, 180 indicates the number of training samples (few-shot).
we have 30 classes => 30 indicates 1 sample per class!

with-pretrain, without-pretrain-pointtransformer, without-pretrain-pointnet indicates different neural net.

we need each neural net's "BEST" Val/Acc on different #training sample configuration.
"""

import pandas as pd
import os
import argparse

def get_stats(ignore_shots=None):
    csv_path = os.path.join(os.path.dirname(__file__), 'cls-few-shot.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for 'Val/Acc' metric
    # Check if 'metric_key' exists, otherwise try 'metric' for backward compatibility or different format
    if 'metric_key' in df.columns:
        metric_col = 'metric_key'
    elif 'metric' in df.columns:
        metric_col = 'metric'
    else:
        print("Error: Could not find metric column (expected 'metric_key' or 'metric')")
        return

    df = df[df[metric_col] == 'Val/Acc']

    # Identify run name column
    if 'run_name' in df.columns:
        run_name_col = 'run_name'
    elif 'Run' in df.columns:
        run_name_col = 'Run'
    else:
        print("Error: Could not find run name column (expected 'run_name' or 'Run')")
        return
        
    # Identify run id column
    if 'run_id' in df.columns:
        run_id_col = 'run_id'
    elif 'Run ID' in df.columns:
        run_id_col = 'Run ID'
    else:
        # If no explicit ID, use name as ID (though risk of collision if same name used multiple times)
        run_id_col = run_name_col

    # Function to parse Run name
    def parse_run(run_name):
        if not isinstance(run_name, str):
             return "unknown", -1
        parts = run_name.rsplit('-', 1)
        if len(parts) == 2 and parts[1].isdigit():
            model = parts[0]
            shots = int(parts[1])
            return model, shots
        return run_name, -1

    # Apply parsing
    df[['Model', 'Shots']] = df[run_name_col].apply(lambda x: pd.Series(parse_run(x)))

    # Filter ignored shots
    if ignore_shots:
        print(f"Ignoring shots: {ignore_shots}")
        df = df[~df['Shots'].isin(ignore_shots)]

    # Calculate Best Val/Acc and Last Val/Acc for each Run ID
    # First, ensure data is sorted by step to correctly identify the last value
    df = df.sort_values(by=[run_id_col, 'step'])
    
    # Group by Run ID to get metrics for each run
    run_stats = df.groupby([run_id_col, 'Model', 'Shots']).agg(
        Best_Val_Acc=('value', 'max'),
        Last_Val_Acc=('value', 'last')
    ).reset_index()

    # Now aggregate by Model and Shots to get Mean across different seeds/runs
    # User requested removing STD, so we keep mean and count
    stats = run_stats.groupby(['Model', 'Shots'])[['Best_Val_Acc', 'Last_Val_Acc']].agg(['mean', 'count']).reset_index()

    # Flatten the multi-level column index for cleaner output
    stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in stats.columns.values]
    
    # Calculate Shots Per Class (assuming 30 classes as per comments)
    stats['Shots_Per_Class'] = stats['Shots'] / 30

    # Sort for better readability
    stats = stats.sort_values(by=['Model', 'Shots'])

    # Format the output
    print("Aggregation Results (Best and Last Val/Acc):")
    print(stats.to_string(index=False))
    
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate classification statistics.")
    parser.add_argument('--ignore-shot', type=int, nargs='+', help='List of shots to ignore (e.g. 60 90)')
    args = parser.parse_args()
    get_stats(ignore_shots=args.ignore_shot)
