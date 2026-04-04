#!/usr/bin/env python3
"""Collect statistics from inference results across all samples."""

from pathlib import Path
import json
import numpy as np
from argparse import ArgumentParser


def collect_statistics(data_dir: Path) -> dict:
    """Collect statistics from all samples.
    
    Args:
        data_dir: Directory containing sample folders with inferred/results.json
        
    Returns:
        Dictionary mapping precision name to statistics.
    """
    all_precision_stats = {}
    
    sample_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(len(sample_dirs))
    
    for sample_dir in sample_dirs:
        results_file = sample_dir / "inferred" / "fastspectrum_results.json"
        if not results_file.exists():
            print(f"Warning: {results_file} does not exist, skipping.")
            continue
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        precisions = results.get("precisions")
        if precisions:
            # New format with multiple precisions
            for p_name, p_data in precisions.items():
                if p_name not in all_precision_stats:
                    all_precision_stats[p_name] = {
                        "subspace_losses": [],
                        "pointcloud_vs_network_means": [],
                        "eval_relerr_means": [],
                        "qr_times": [],
                        "forward_times": [],
                        "network_gev_times": [],
                        "pointcloud_gev_times": [],
                        "same_residual_gev_times": [],
                    }
                
                scores = p_data.get("scores", {})
                times = p_data.get("times", {})
                
                subspace_loss = p_data.get("loss")
                pointcloud_vs_network = scores.get("pointcloud_vs_network")
                if pointcloud_vs_network is None:
                    pointcloud_vs_network = scores.get("mesh_vs_fastspectrum")
                else:
                    print(f"Warning: {results_file} does not contain mesh_vs_fastspectrum, skipping.")
                eval_relerr = scores.get("eval_relerr")
                
                if subspace_loss is not None:
                    all_precision_stats[p_name]["subspace_losses"].append(subspace_loss)
                if pointcloud_vs_network is not None:
                    all_precision_stats[p_name]["pointcloud_vs_network_means"].append(np.mean(pointcloud_vs_network))
                if eval_relerr is not None:
                    all_precision_stats[p_name]["eval_relerr_means"].append(np.mean(eval_relerr))
                
                if "qr" in times: all_precision_stats[p_name]["qr_times"].append(times["qr"])
                if "forward" in times: all_precision_stats[p_name]["forward_times"].append(times["forward"])
                if "fastspectrum" in times: all_precision_stats[p_name]["network_gev_times"].append(times["fastspectrum"])
                if "pointcloud_gev" in times: all_precision_stats[p_name]["pointcloud_gev_times"].append(times["pointcloud_gev"])
                if "same_residual_gev" in times: all_precision_stats[p_name]["same_residual_gev_times"].append(times["same_residual_gev"])
        else:
            # Old format or fallback
            p_name = "default"
            if p_name not in all_precision_stats:
                all_precision_stats[p_name] = {
                    "subspace_losses": [],
                    "pointcloud_vs_network_means": [],
                    "eval_relerr_means": [],
                    "qr_times": [],
                    "forward_times": [],
                    "network_gev_times": [],
                    "pointcloud_gev_times": [],
                    "same_residual_gev_times": [],
                }
            
            scores = results.get("scores", {})
            times = results.get("times", {})

            subspace_loss = scores.get("subspace_loss")
            pointcloud_vs_network = scores.get("pointcloud_vs_network")
            if pointcloud_vs_network is None:
                pointcloud_vs_network = scores.get("mesh_vs_fastspectrum")
            eval_relerr = scores.get("eval_relerr")
            
            if subspace_loss is not None:
                all_precision_stats[p_name]["subspace_losses"].append(subspace_loss)
            if pointcloud_vs_network is not None:
                all_precision_stats[p_name]["pointcloud_vs_network_means"].append(np.mean(pointcloud_vs_network))
            if eval_relerr is not None:
                all_precision_stats[p_name]["eval_relerr_means"].append(np.mean(eval_relerr))

            if "qr" in times: all_precision_stats[p_name]["qr_times"].append(times["qr"])
            if "forward" in times: all_precision_stats[p_name]["forward_times"].append(times["forward"])
            if "fastspectrum" in times: all_precision_stats[p_name]["network_gev_times"].append(times["fastspectrum"])
            if "pointcloud_gev" in times: all_precision_stats[p_name]["pointcloud_gev_times"].append(times["pointcloud_gev"])
            if "same_residual_gev" in times: all_precision_stats[p_name]["same_residual_gev_times"].append(times["same_residual_gev"])

    def get_stats(data):
        if data is None or (isinstance(data, (list, np.ndarray)) and len(data) == 0):
            return None
        data = np.array(data)
        return {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "values": data.tolist(),
        }

    final_results = {}
    for p_name, data in all_precision_stats.items():
        if not data["subspace_losses"]:
            continue
        
        final_results[p_name] = {
            "subspace_loss": get_stats(data["subspace_losses"]),
            "pointcloud_vs_network": get_stats(1 - np.array(data["pointcloud_vs_network_means"])) if data["pointcloud_vs_network_means"] else None,
            "eval_relerr": get_stats(data["eval_relerr_means"]),
            "qr_time": get_stats(data["qr_times"]),
            "forward_time": get_stats(data["forward_times"]),
            "network_gev_time": get_stats(data["network_gev_times"]),
            "pointcloud_gev_time": get_stats(data["pointcloud_gev_times"]),
            "same_residual_gev_time": get_stats(data["same_residual_gev_times"]),
            "num_samples": len(data["subspace_losses"]),
        }
    
    return final_results


def main():
    parser = ArgumentParser(description="Collect statistics from inference results.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing sample folders.")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    all_stats = collect_statistics(data_dir)
    
    for p_name, stats in all_stats.items():
        print(f"\n" + "="*40)
        print(f" Statistics for {p_name} from {stats['num_samples']} samples:")
        print("="*40)
        
        if stats['subspace_loss']:
            print(f"\nsubspace_loss:")
            print(f"  mean: {stats['subspace_loss']['mean']:.6f}")
            print(f"  std:  {stats['subspace_loss']['std']:.6f}")
            print(f"  median: {stats['subspace_loss']['median']:.6f}")
        
        if stats['pointcloud_vs_network']:
            print(f"\nmean(pointcloud_vs_network):")
            print(f"  mean: {stats['pointcloud_vs_network']['mean']:.6f}")
            print(f"  std:  {stats['pointcloud_vs_network']['std']:.6f}")
            print(f"  median: {stats['pointcloud_vs_network']['median']:.6f}")
        
        if stats['eval_relerr']:
            print(f"\nmean(eval_relerr):")
            print(f"  mean: {stats['eval_relerr']['mean']:.6f}")
            print(f"  std:  {stats['eval_relerr']['std']:.6f}")
            print(f"  median: {stats['eval_relerr']['median']:.6f}")

        time_fields = [
            ('qr_time', 'QR decomposition time'),
            ('forward_time', 'Forward pass time'),
            ('network_gev_time', 'Network GEV time'),
            ('pointcloud_gev_time', 'Pointcloud GEV time'),
            ('same_residual_gev_time', 'Same residual GEV time'),
        ]

        for field, label in time_fields:
            if stats.get(field):
                print(f"\n{label}:")
                print(f"  mean: {stats[field]['mean']:.6f}s")
                print(f"  std:  {stats[field]['std']:.6f}s")
                print(f"  median: {stats[field]['median']:.6f}s")
    
    output_file = data_dir / "statistics.json"
    with open(output_file, 'w') as f:
        json.dump(all_stats, f, indent=4)
    print(f"\nStatistics saved to: {output_file}")



if __name__ == "__main__":
    main()
