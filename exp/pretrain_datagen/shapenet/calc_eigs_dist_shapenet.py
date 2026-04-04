from g2pt.sft import SelfSupervisedDataset, balance_stiffness
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser
import numpy as np
from scipy import sparse as sp
from scipy.linalg import eigvalsh
import torch


def calc_min_max_ratio(data, delta:float, k:int) -> float:
    """Calculate the ratio of the minimum to the maximum eigenvalue of the stiffness matrix.

    Args:
        stiff (sp.coo_matrix): stiffness matrix.

    Returns:
        float: ratio of the minimum to the maximum eigenvalue.
    """
    stiff = data['stiffness'].astype(np.float64)
    mass = data['mass'].numpy().astype(np.float64)
    mass = sp.diags(mass.reshape(-1))
    L, M = balance_stiffness(stiff, mass, delta, k)
    eigs = eigvalsh(L.todense(), M.todense())

    smallest = eigs[0]
    largest = eigs[k - 1]
    return largest / smallest


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/data/ShapeNetCore.v2/")
    parser.add_argument("--n_samples", type=int, default=10, help="0 for all samples, >0 for limited")
    parser.add_argument("--target_k", type=int, default=96)
    parser.add_argument("--downsample", type=int, default=1024, help="Downsample the point cloud to this number of points.")
    parser.add_argument("--delta", type=float, default=10.0, help="Balance factor for the stiffness matrix.")
    parser.add_argument("--enable_rotate", type=int, default=0, help="Enable rotation augmentation.")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel processes.")
    args = parser.parse_args()
    print(args)
    dataset = SelfSupervisedDataset(
        n_points=args.downsample,
        dataset_name="shapenet",
        dataset_base_path=args.base_path,
        enable_rotate=args.enable_rotate,
        target_k=args.target_k,
        delta=args.delta,
    )

    total = len(dataset)
    if args.n_samples > 0:
        total = min(total, args.n_samples)
    print(f"Total samples to compute: {total}")

    parallism = args.parallel
    if parallism == 0:
        parallism = cpu_count()
    print(f"Using {parallism} parallel processes.")

    if parallism == 1:
        results = [calc_min_max_ratio(dataset[i], args.delta, args.target_k) for i in range(total)]
    else:
        with Pool(processes=parallism) as pool:
            print(f"Mapping {total} samples to {parallism} processes.")
            results = pool.starmap(
                calc_min_max_ratio,
                zip(dataset, [args.delta] * total, [args.target_k] * total),
            )

    print(f"Min/Max ratio: mean={np.mean(results):.4e}, std={np.std(results):.4e}")

if __name__ == "__main__":
    main()