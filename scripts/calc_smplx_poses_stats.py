import argparse
from pathlib import Path

import h5py
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Calculate mean and std of poses in SMPLX deformation dataset.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the HDF5 data file.")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        print(f"Error: Data file {args.data_file} does not exist.")
        return

    print(f"Opening HDF5 file: {args.data_file}...")
    with h5py.File(data_path, "r") as hf:
        pose_keys = ["tar_poses", "src_poses", "poses"]
        found_key = None
        for key in pose_keys:
            if key in hf:
                found_key = key
                print(f"Detected pose key: {found_key}")
                
                poses_ds = hf[found_key]
                all_poses = np.array(poses_ds, dtype=np.float32)
                
                # Check if they are dummy
                if all_poses.ndim == 1 or (all_poses.shape[1] == 1 and np.all(all_poses == 0)):
                    print(f"  Skipping {found_key} (dummy or invalid).")
                    continue

                print(f"  Calculating statistics for {len(all_poses)} samples...")
                mean_poses = np.mean(all_poses, axis=0)
                std_poses = np.std(all_poses, axis=0)

                print("\n" + "="*50)
                print(f"Pose Statistics for {data_path.name} [{found_key}]")
                print("="*50)
                print(f"Number of samples: {len(all_poses)}")
                print(f"Pose dimensions:   {all_poses.shape[1]}")
                print("-" * 50)
                print(f"Mean Pose:\n{mean_poses}")
                print("-" * 50)
                print(f"Std Pose:\n{std_poses}")
                print("="*50)

                # Also print as a list for easy copy-pasting
                print(f"\nMean {found_key} (list format):")
                print(mean_poses.tolist())
                print(f"\nStd {found_key} (list format):")
                print(std_poses.tolist())
                print("\n")
        
        if found_key is None:
            print(f"Error: No pose datasets (tar_poses, src_poses, or poses) found in {args.data_file}")

if __name__ == "__main__":
    main()
