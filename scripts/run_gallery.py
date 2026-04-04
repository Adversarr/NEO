"""Model-Mesh gallery orchestration script."""
import argparse
import subprocess
import shutil
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate mesh sampling and model inference for gallery.")
    parser.add_argument("--mesh_dir", type=str, default="ldata/featuring_models", help="Directory containing mesh files.")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/Finals", help="Directory containing model checkpoints.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 2048, 8192, 32768, 131072], help="Resolutions to sample.")
    parser.add_argument("--k", type=int, default=128, help="Number of eigenfunctions to compute.")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Main temporary directory.")
    return parser.parse_args()

def run_command(cmd, description):
    print(f"🚀 Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Finished: {description}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during: {description}")
        print(e)
        return False
    return True

def main():
    args = parse_args()
    mesh_dir = Path(args.mesh_dir)
    ckpt_dir = Path(args.ckpt_dir)
    tmp_root = Path(args.tmp_dir)
    tmp_root.mkdir(exist_ok=True)
    
    tmp_base = tmp_root / "base_sampling"
    
    # 1. Multi-resolution sampling
    print("--- Phase 1: Multi-resolution Sampling ---")
    sampling_cmd = [
        "python", "scripts/multi_sampling.py",
        "--mesh_dir", str(mesh_dir),
        "--out_dir", str(tmp_base),
        "--n_points"
    ] + [str(r) for r in args.resolutions] + [
        "--k", str(args.k),
        "--normalize"
    ]
    
    if not run_command(sampling_cmd, "Multi-resolution sampling"):
        return

    # 2. Inference for each checkpoint and each mesh
    print("\n--- Phase 2: Inference for each Checkpoint and Mesh ---")
    ckpt_files = sorted(list(ckpt_dir.glob("*.ckpt")))
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    # Get list of meshes from the base sampling directory
    meshes = sorted([d.name for d in tmp_base.iterdir() if d.is_dir()])
    
    for ckpt_path in ckpt_files:
        ckpt_name = ckpt_path.stem
        
        for mesh_name in meshes:
            target_dir = tmp_root / f"tmp_{ckpt_name}_{mesh_name}"
            
            print(f"\nProcessing: CKPT={ckpt_name}, MESH={mesh_name}")
            
            # Copy sampled data for this specific mesh to target_dir
            if target_dir.exists():
                print(f"  Removing existing directory: {target_dir}")
                shutil.rmtree(target_dir)
            
            target_dir.mkdir(parents=True)
            
            mesh_base_dir = tmp_base / mesh_name
            for res_dir in mesh_base_dir.iterdir():
                if res_dir.is_dir():
                    shutil.copytree(res_dir, target_dir / res_dir.name)
            
            # Run inference
            # data_dir is tmp/tmp_CKPT_MESH, so glob "*" matches resolutions
            infer_cmd = [
                "python", "exp/pretrain/infer.py",
                "--ckpt", str(ckpt_path),
                "--data_dir", str(target_dir),
                "--glob", "*",
                "--k", str(args.k)
            ]
            
            run_command(infer_cmd, f"Inference for {ckpt_name} on {mesh_name}")

    print("\n🎉 Gallery pipeline completed!")

if __name__ == "__main__":
    main()
