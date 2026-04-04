import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_pretrain", type=str, required=True,
                       help="Path to pretrained checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Directory to save JSON outputs")
    parser.add_argument("--cases", type=str, nargs="+", default=None,
                       help="Specific cases to run (e.g., 'Bed-1 Chair-1'). If not provided, runs all cases")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel processes")
    parser.add_argument("--fast_dev_run", action="store_true",
                       help="Run fast dev run for testing")
    return parser.parse_args()

def run_model_command(cmd, model_name):
    """Helper function to run a model training command"""
    try:
        print(f"Starting {model_name} model...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"{model_name} model completed successfully")
        print(f"Output (first 500 chars): {result.stdout[:500]}...")
        if result.stderr:
            print(f"Stderr (first 500 chars): {result.stderr[:500]}...")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{model_name} model failed!")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error code: {e.returncode}")
        if e.stdout:
            print(f"Stdout (last 500 chars): {e.stdout[-500:] if len(e.stdout) > 500 else e.stdout}")
        if e.stderr:
            print(f"Stderr (last 500 chars): {e.stderr[-500:] if len(e.stderr) > 500 else e.stderr}")
        return False

def run_single(case: str, ckpt_pretrain: str, output_dir: str, fast_dev_run: bool = False):
    """Run both pretrained and non-pretrained models for a single case using subprocesses"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build command for pretrained model
    export_json_pretrained = os.path.join(output_dir, f"{case}-pretrained.json")
    cmd_pretrained = [
        "python", "exp/downstream/seg.py",
        "datamod=partnet_h5",
        f"datamod.case={case}",
        f"ckpt_pretrain={ckpt_pretrain}",
        "freeze_pretrained=true",
        f"fast_dev_run={'true' if fast_dev_run else 'false'}",
        f"+export_json={export_json_pretrained}"
    ]

    # Build command for non-pretrained model (only_pos_embed)
    export_json_no_pretrained = os.path.join(output_dir, f"{case}-no-pretrained.json")
    cmd_no_pretrained = [
        "python", "exp/downstream/seg.py",
        "datamod=partnet_h5",
        f"datamod.case={case}",
        "model=only_pos_embed",
        f"fast_dev_run={'true' if fast_dev_run else 'false'}",
        f"+export_json={export_json_no_pretrained}"
    ]

    print(f"\n{'='*80}")
    print(f"Running case: {case}")
    print(f"Pretrained command: {' '.join(cmd_pretrained)}")
    print(f"No-pretrained command: {' '.join(cmd_no_pretrained)}")
    print(f"{'='*80}\n")

    # Run both commands sequentially (could be changed to parallel if needed)
    success_pretrained = run_model_command(cmd_pretrained, "pretrained")
    success_no_pretrained = run_model_command(cmd_no_pretrained, "non-pretrained")

    if success_pretrained and success_no_pretrained:
        print(f"✓ Both models completed successfully for case: {case}")
        return True
    else:
        print(f"✗ One or both models failed for case: {case}")
        return False

def find_all_cases(root_dir):
    """
    > ls data/processed_sgpn/ | head
    train-Bag-1-00_pc.hdf5
    train-Bed-1-00_pc.hdf5
    train-Bed-2-00_pc.hdf5
    train-Bed-3-00_pc.hdf5
    train-Bottle-1-00_pc.hdf5
    train-Bottle-3-00_pc.hdf5
    train-Bowl-1-00_pc.hdf5
    train-Chair-1-00_pc.hdf5
    train-Chair-1-01_pc.hdf5
    train-Chair-1-02_pc.hdf5"""
    cases = set()

    if not os.path.exists(root_dir):
        print(f"Warning: Directory {root_dir} does not exist")
        return []

    for filename in os.listdir(root_dir):
        if filename.startswith("train-") and filename.endswith("_pc.hdf5"):
            # Extract case name: train-Bed-1-00_pc.hdf5 -> Bed-1
            # Remove 'train-' prefix and split by '-'
            parts = filename[6:].split('-')  # Remove 'train-'
            if len(parts) >= 2:
                # Case name is first two parts: Bed-1 from ['Bed', '1', '00_pc.hdf5']
                case_name = f"{parts[0]}-{parts[1]}"
                cases.add(case_name)

    return sorted(list(cases))

def main():
    args = parse_args()

    # Find all cases
    data_dir = "data/processed_sgpn"
    all_cases = find_all_cases(data_dir)

    if not all_cases:
        print(f"No cases found in {data_dir}")
        return

    # Filter cases if specific cases are provided
    if args.cases:
        cases_to_run = [case for case in args.cases if case in all_cases]
        not_found = [case for case in args.cases if case not in all_cases]
        if not_found:
            print(f"Warning: Some cases not found: {not_found}")
        if not cases_to_run:
            print("No valid cases to run")
            return
    else:
        cases_to_run = all_cases

    print(f"Found {len(all_cases)} total cases")
    print(f"Running {len(cases_to_run)} cases: {cases_to_run}")

    # Use ProcessPoolExecutor for better result handling
    success_cases = []
    failed_cases = []

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_case = {}
        for case in cases_to_run:
            future = executor.submit(
                run_single, case, args.ckpt_pretrain, args.output_dir, args.fast_dev_run
            )
            future_to_case[future] = case

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_case):
            case = future_to_case[future]
            completed += 1

            try:
                success = future.result(timeout=1)
                if success:
                    success_cases.append(case)
                    print(f"[{completed}/{len(cases_to_run)}] ✓ Case {case}: SUCCESS")
                else:
                    failed_cases.append(case)
                    print(f"[{completed}/{len(cases_to_run)}] ✗ Case {case}: FAILED")
            except Exception as e:
                failed_cases.append(case)
                print(f"[{completed}/{len(cases_to_run)}] ✗ Case {case}: ERROR - {e}")

    # Print summary
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"Total cases: {len(cases_to_run)}")
    print(f"Successful: {len(success_cases)}")
    print(f"Failed: {len(failed_cases)}")

    if success_cases:
        print(f"\nSuccessful cases: {success_cases}")

    if failed_cases:
        print(f"\nFailed cases: {failed_cases}")
        print("\nCheck the logs above for detailed error messages.")

if __name__ == '__main__':
    main()