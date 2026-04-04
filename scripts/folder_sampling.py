from pathlib import Path
from argparse import ArgumentParser
import subprocess
import sys

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Input folder containing mesh files.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder to save results.")
    parser.add_argument("--n_points", type=int, default=1024, help="Number of points to sample.")
    parser.add_argument("--subdiv", type=int, default=0, help="Number of times to subdivide the mesh.")
    parser.add_argument("--k", type=int, default=96, help="Number of mesh eigenfunctions to compute.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--no_uniform", action="store_true", help="Disable uniform sampling.")
    parser.add_argument("--non_uniform_sigmas", type=float, nargs='+', default=[], help="Scales for non-uniform sampling.")
    parser.add_argument("--normalize", action="store_true", help="Normalize the points to the unit sphere.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.input.exists():
        print(f"Error: Input directory {args.input} does not exist.")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    
    # Supported extensions based on sample_points.py load_mesh
    extensions = {".obj", ".ply", ".stl", ".off", ".msh"}
    mesh_files = [f for f in args.input.iterdir() if f.suffix.lower() in extensions]
    
    if not mesh_files:
        print(f"No mesh files found in {args.input}")
        return

    sample_script = Path(__file__).parent / "sample_points.py"
    if not sample_script.exists():
        print(f"Error: {sample_script} not found.")
        return

    for mesh_file in mesh_files:
        obj_name = mesh_file.stem
        print(f"Processing {obj_name}...")
        
        # 1. Uniform sampling
        # Folder: output/ObjName-Uniform
        if not args.no_uniform:
            uniform_out_dir = args.output / f"{obj_name}-Uniform"
            uniform_out_dir.mkdir(parents=True, exist_ok=True)
            uniform_out_file = uniform_out_dir / "sample_points.npy"
            
            cmd_uniform = [
                sys.executable, str(sample_script),
                "--mesh_path", str(mesh_file),
                "--out", str(uniform_out_file),
                "--n_points", str(args.n_points),
                "--subdiv", str(args.subdiv),
                "--k", str(args.k)
            ]
            if args.seed is not None:
                cmd_uniform.extend(["--seed", str(args.seed)])
            if args.normalize:
                cmd_uniform.append("--normalize")
                
            print(f"  Running Uniform sampling -> {uniform_out_dir.name}")
            subprocess.run(cmd_uniform, check=True)

        # 2. Non-Uniform sampling (multiple sigmas)
        for sigma in args.non_uniform_sigmas:
            # Folder: output/ObjName-NonUniform-{sigma}
            # We use a naming convention that includes the sigma value
            folder_name = f"{obj_name}-NonUniform-{sigma}"
            non_uniform_out_dir = args.output / folder_name
            non_uniform_out_dir.mkdir(parents=True, exist_ok=True)
            non_uniform_out_file = non_uniform_out_dir / "sample_points.npy"
            
            cmd_non_uniform = [
                sys.executable, str(sample_script),
                "--mesh_path", str(mesh_file),
                "--out", str(non_uniform_out_file),
                "--n_points", str(args.n_points),
                "--subdiv", str(args.subdiv),
                "--k", str(args.k),
                "--non_uniform",
                "--non_uniform_sigma", str(sigma)
            ]
            if args.seed is not None:
                cmd_non_uniform.extend(["--seed", str(args.seed)])
            if args.normalize:
                cmd_non_uniform.append("--normalize")
                
            print(f"  Running Non-Uniform sampling (sigma={sigma}) -> {folder_name}")
            subprocess.run(cmd_non_uniform, check=True)

if __name__ == "__main__":
    main()
