import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Case:
    ckpt: str
    mesh: str
    n_points: int
    data: dict[str, Any]
    json_path: str


def _discover_mesh_names(root_dir: Path) -> list[str]:
    base = root_dir / "base_sampling"
    if not base.exists():
        return []
    names = [p.name for p in base.iterdir() if p.is_dir()]
    names.sort(key=lambda s: (-len(s), s))
    return names


def _parse_tmp_dir_name(dir_name: str, mesh_names: list[str]) -> tuple[str, str] | None:
    if not dir_name.startswith("tmp_"):
        return None
    rest = dir_name[len("tmp_") :]
    if not rest:
        return None

    for mesh in mesh_names:
        suffix = f"_{mesh}"
        if rest.endswith(suffix):
            ckpt = rest[: -len(suffix)]
            if ckpt:
                return ckpt, mesh
            return None

    parts = rest.split("_")
    if len(parts) < 2:
        return None
    ckpt = "_".join(parts[:-1])
    mesh = parts[-1]
    if not ckpt or not mesh:
        return None
    return ckpt, mesh


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _iter_cases(root_dir: Path) -> Iterable[Case]:
    mesh_names = _discover_mesh_names(root_dir)
    for tmp_dir in sorted(root_dir.iterdir()):
        if not tmp_dir.is_dir():
            continue
        if tmp_dir.name == "base_sampling":
            continue
        parsed = _parse_tmp_dir_name(tmp_dir.name, mesh_names)
        if parsed is None:
            continue
        ckpt, mesh = parsed

        for n_dir in sorted(tmp_dir.iterdir(), key=lambda p: p.name):
            if not n_dir.is_dir():
                continue
            try:
                n_points = int(n_dir.name)
            except Exception:
                continue

            json_path = n_dir / "inferred" / "results.json"
            if not json_path.exists():
                json_path = n_dir / "results.json"
            if not json_path.exists():
                continue

            data = _load_json(json_path)
            if data is None:
                continue

            n_points_data = data.get("n_points")
            if isinstance(n_points_data, int) and n_points_data > 0:
                n_points = n_points_data

            yield Case(
                ckpt=ckpt,
                mesh=mesh,
                n_points=int(n_points),
                data=data,
                json_path=json_path.as_posix(),
            )


def _percentiles(x: np.ndarray, ps: list[float]) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    vals = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, vals)}


def _get_scores(data: dict[str, Any], key: str) -> np.ndarray | None:
    scores = data.get("scores", {}).get(key)
    if scores is None:
        scores = data.get("precisions", {}).get("fp32", {}).get("scores", {}).get(key)
    if scores is None:
        return None
    return np.asarray(scores, dtype=np.float64)


def _get_scores_precision(data: dict[str, Any], precision: str, key: str) -> np.ndarray | None:
    scores = data.get("precisions", {}).get(precision, {}).get("scores", {}).get(key)
    if scores is None:
        return None
    return np.asarray(scores, dtype=np.float64)


def _get_times(data: dict[str, Any]) -> dict[str, float] | None:
    times = data.get("times")
    if not isinstance(times, dict):
        return None
    out: dict[str, float] = {}
    for k in ["forward", "network_gev", "pointcloud_gev"]:
        v = times.get(k)
        if v is None:
            return None
        out[k] = float(v)
    return out


def _get_times_precision(data: dict[str, Any], precision: str) -> dict[str, float] | None:
    times = data.get("precisions", {}).get(precision, {}).get("times")
    if not isinstance(times, dict):
        return None
    out: dict[str, float] = {}
    for k in ["forward", "network_gev", "pointcloud_gev"]:
        v = times.get(k)
        if v is None:
            return None
        out[k] = float(v)
    return out


def _pick_n_target(cases: list[Case], n_target: int | None) -> int | None:
    if n_target is not None:
        return int(n_target)
    if not cases:
        return None
    return int(max(c.n_points for c in cases))


def compute_runtime_stats(
    cases: list[Case],
    model: str,
    n_target: int | None,
) -> dict[str, Any]:
    model_cases = [c for c in cases if c.ckpt == model]
    n_target_picked = _pick_n_target(model_cases, n_target)
    if n_target_picked is None:
        return {"model": model, "n_target": None, "error": "no cases found"}

    model_cases = [c for c in model_cases if c.n_points == n_target_picked]
    if not model_cases:
        return {"model": model, "n_target": n_target_picked, "error": "no cases at n_target"}

    speedups_total = []
    speedups_forward = []
    per_mesh = {}

    for c in model_cases:
        times = _get_times(c.data)
        if times is None:
            continue
        arpack = times["pointcloud_gev"]
        forward = times["forward"]
        total = times["forward"] + times["network_gev"]
        if arpack <= 0.0 or total <= 0.0 or forward <= 0.0:
            continue

        speedups_total.append(arpack / total)
        speedups_forward.append(arpack / forward)
        per_mesh[c.mesh] = {
            "arpack_s": float(arpack),
            "NEO_total_s": float(total),
            "NEO_forward_s": float(forward),
            "speedup_total_x": float(arpack / total),
            "speedup_forward_x": float(arpack / forward),
        }

    speedups_total_arr = np.asarray(speedups_total, dtype=np.float64)
    speedups_forward_arr = np.asarray(speedups_forward, dtype=np.float64)

    return {
        "model": model,
        "n_target": int(n_target_picked),
        "speedup_total": {
            **_percentiles(speedups_total_arr, [10, 50, 90]),
            "mean": float(np.mean(speedups_total_arr)) if speedups_total_arr.size else float("nan"),
        },
        "speedup_forward": {
            **_percentiles(speedups_forward_arr, [10, 50, 90]),
            "mean": float(np.mean(speedups_forward_arr)) if speedups_forward_arr.size else float("nan"),
        },
        "per_mesh": per_mesh,
    }


def compute_crossover_stats(
    cases: list[Case],
    model: str,
) -> dict[str, Any]:
    model_cases = [c for c in cases if c.ckpt == model]
    if not model_cases:
        return {"model": model, "error": "no cases found"}

    per_mesh: dict[str, dict[int, dict[str, float]]] = {}
    for c in model_cases:
        times = _get_times(c.data)
        if times is None:
            continue
        per_mesh.setdefault(c.mesh, {})[int(c.n_points)] = {
            "arpack_s": float(times["pointcloud_gev"]),
            "NEO_total_s": float(times["forward"] + times["network_gev"]),
        }

    crossovers = []
    per_mesh_crossover = {}
    for mesh, entries in per_mesh.items():
        ns = sorted(entries.keys())
        crossover_n = None
        for n in ns:
            arpack = entries[n]["arpack_s"]
            NEO = entries[n]["NEO_total_s"]
            if arpack > 0.0 and NEO > 0.0 and NEO <= arpack:
                crossover_n = int(n)
                break
        per_mesh_crossover[mesh] = crossover_n
        if crossover_n is not None:
            crossovers.append(crossover_n)

    crossover_arr = np.asarray(crossovers, dtype=np.float64)
    return {
        "model": model,
        "per_mesh": per_mesh_crossover,
        "crossover_n_points": {
            **_percentiles(crossover_arr, [10, 50, 90]),
            "count": int(crossover_arr.size),
        },
    }


def compute_accuracy_stats(
    cases: list[Case],
    model: str,
    n_target: int | None,
    k_eval: int,
    k_low: int,
) -> dict[str, Any]:
    model_cases = [c for c in cases if c.ckpt == model]
    n_target_picked = _pick_n_target(model_cases, n_target)
    if n_target_picked is None:
        return {"model": model, "n_target": None, "error": "no cases found"}

    model_cases = [c for c in model_cases if c.n_points == n_target_picked]
    if not model_cases:
        return {"model": model, "n_target": n_target_picked, "error": "no cases at n_target"}

    mse_all = []
    mse_low = []
    per_case_median = []

    for c in model_cases:
        s = _get_scores(c.data, "pointcloud_vs_network")
        if s is None or s.size == 0:
            continue
        mse = 1.0 - np.clip(s, -1.0, 1.0)
        ke = int(min(k_eval, mse.size))
        kl = int(min(k_low, mse.size))
        mse_all.append(mse[:ke])
        mse_low.append(mse[:kl])
        per_case_median.append(float(np.median(mse[:ke])))

    if not mse_all:
        return {"model": model, "n_target": n_target_picked, "error": "no pointcloud_vs_network scores"}

    mse_all_arr = np.concatenate(mse_all, axis=0).astype(np.float64)
    mse_low_arr = np.concatenate(mse_low, axis=0).astype(np.float64)
    per_case_median_arr = np.asarray(per_case_median, dtype=np.float64)

    frac_median_lt_001 = float(np.mean(per_case_median_arr < 1e-2)) if per_case_median_arr.size else float("nan")

    return {
        "model": model,
        "n_target": int(n_target_picked),
        "score_key": "pointcloud_vs_network",
        "k_eval": int(k_eval),
        "k_low": int(k_low),
        "mse_all_modes": {
            **_percentiles(mse_all_arr, [10, 50, 90, 95, 99]),
            "mean": float(np.mean(mse_all_arr)),
        },
        "mse_low_modes": {
            **_percentiles(mse_low_arr, [10, 50, 90, 95, 99]),
            "mean": float(np.mean(mse_low_arr)),
        },
        "per_case_median_mse": {
            **_percentiles(per_case_median_arr, [10, 50, 90]),
            "mean": float(np.mean(per_case_median_arr)),
            "frac_lt_1e-2": frac_median_lt_001,
            "count": int(per_case_median_arr.size),
        },
    }


def compute_mixed_precision_stats(
    cases: list[Case],
    model: str,
    n_target: int | None,
    k_eval: int,
) -> dict[str, Any]:
    model_cases = [c for c in cases if c.ckpt == model]
    n_target_picked = _pick_n_target(model_cases, n_target)
    if n_target_picked is None:
        return {"model": model, "n_target": None, "error": "no cases found"}

    model_cases = [c for c in model_cases if c.n_points == n_target_picked]
    if not model_cases:
        return {"model": model, "n_target": n_target_picked, "error": "no cases at n_target"}

    mse_delta = []
    forward_speedups = []
    total_speedups = []

    for c in model_cases:
        s32 = _get_scores_precision(c.data, "fp32", "pointcloud_vs_network")
        s16 = _get_scores_precision(c.data, "fp16", "pointcloud_vs_network")
        t32 = _get_times_precision(c.data, "fp32")
        t16 = _get_times_precision(c.data, "fp16")
        if s32 is None or s16 is None or t32 is None or t16 is None:
            continue

        k = int(min(k_eval, s32.size, s16.size))
        if k <= 0:
            continue

        mse32 = 1.0 - np.clip(s32[:k], -1.0, 1.0)
        mse16 = 1.0 - np.clip(s16[:k], -1.0, 1.0)
        mse_delta.append(mse16 - mse32)

        f32 = float(t32["forward"])
        f16 = float(t16["forward"])
        tot32 = float(t32["forward"] + t32["network_gev"])
        tot16 = float(t16["forward"] + t16["network_gev"])
        if f32 > 0.0 and f16 > 0.0:
            forward_speedups.append(f32 / f16)
        if tot32 > 0.0 and tot16 > 0.0:
            total_speedups.append(tot32 / tot16)

    if not mse_delta:
        return {"model": model, "n_target": n_target_picked, "error": "no fp16/fp32 paired data"}

    mse_delta_arr = np.concatenate(mse_delta, axis=0).astype(np.float64)
    forward_speedups_arr = np.asarray(forward_speedups, dtype=np.float64)
    total_speedups_arr = np.asarray(total_speedups, dtype=np.float64)

    return {
        "model": model,
        "n_target": int(n_target_picked),
        "score_key": "pointcloud_vs_network",
        "k_eval": int(k_eval),
        "mse_fp16_minus_fp32": {
            **_percentiles(mse_delta_arr, [10, 50, 90, 95, 99]),
            "mean": float(np.mean(mse_delta_arr)),
        },
        "forward_time_speedup_fp16_over_fp32": {
            **_percentiles(forward_speedups_arr, [10, 50, 90]),
            "mean": float(np.mean(forward_speedups_arr)) if forward_speedups_arr.size else float("nan"),
            "count": int(forward_speedups_arr.size),
        },
        "total_time_speedup_fp16_over_fp32": {
            **_percentiles(total_speedups_arr, [10, 50, 90]),
            "mean": float(np.mean(total_speedups_arr)) if total_speedups_arr.size else float("nan"),
            "count": int(total_speedups_arr.size),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default="ldata/eval_output_full")
    p.add_argument("--model", type=str, default="Ours-Large")
    p.add_argument("--n_target", type=int, default=None)
    p.add_argument("--k_eval", type=int, default=96)
    p.add_argument("--k_low", type=int, default=20)
    p.add_argument("--json_only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir)
    if not root_dir.is_absolute():
        root_dir = (Path.cwd() / root_dir).resolve()

    cases = list(_iter_cases(root_dir))

    out = {
        "root_dir": root_dir.as_posix(),
        "case_count": int(len(cases)),
        "model": args.model,
        "runtime": compute_runtime_stats(cases, args.model, args.n_target),
        "crossover": compute_crossover_stats(cases, args.model),
        "accuracy": compute_accuracy_stats(cases, args.model, args.n_target, args.k_eval, args.k_low),
        "mixed_precision": compute_mixed_precision_stats(cases, args.model, args.n_target, args.k_eval),
    }

    print(json.dumps(out, indent=2))

    if args.json_only:
        return

    runtime = out["runtime"]
    crossover = out["crossover"]
    accuracy = out["accuracy"]
    mp = out["mixed_precision"]

    def fmt(x: Any) -> str:
        try:
            xf = float(x)
        except Exception:
            return str(x)
        if not np.isfinite(xf):
            return "nan"
        if xf >= 100:
            return f"{xf:.1f}"
        if xf >= 10:
            return f"{xf:.2f}"
        if xf >= 1:
            return f"{xf:.3f}"
        if xf >= 1e-2:
            return f"{xf:.4f}"
        return f"{xf:.3e}"

    print()
    if "error" in runtime:
        print(f"[runtime] {runtime['error']}")
    else:
        print(f"[runtime] model={runtime['model']} N={runtime['n_target']}")
        print(
            "  speedup(total=forward+RR): "
            f"p50={fmt(runtime['speedup_total'].get('p50'))}x, "
            f"p10={fmt(runtime['speedup_total'].get('p10'))}x, "
            f"p90={fmt(runtime['speedup_total'].get('p90'))}x"
        )
        print(
            "  speedup(forward-only): "
            f"p50={fmt(runtime['speedup_forward'].get('p50'))}x, "
            f"p10={fmt(runtime['speedup_forward'].get('p10'))}x, "
            f"p90={fmt(runtime['speedup_forward'].get('p90'))}x"
        )

    if "error" in crossover:
        print(f"[crossover] {crossover['error']}")
    else:
        c = crossover["crossover_n_points"]
        print(
            f"[crossover] p50 N={fmt(c.get('p50'))}, p10 N={fmt(c.get('p10'))}, "
            f"p90 N={fmt(c.get('p90'))} (count={c.get('count')})"
        )

    if "error" in accuracy:
        print(f"[accuracy] {accuracy['error']}")
    else:
        a_all = accuracy["mse_all_modes"]
        a_low = accuracy["mse_low_modes"]
        pcm = accuracy["per_case_median_mse"]
        print(f"[accuracy] model={accuracy['model']} N={accuracy['n_target']} k_eval={accuracy['k_eval']}")
        print(
            "  mse(all modes): "
            f"p50={fmt(a_all.get('p50'))}, p90={fmt(a_all.get('p90'))}, p99={fmt(a_all.get('p99'))}"
        )
        print(
            "  mse(k<=k_low): "
            f"p50={fmt(a_low.get('p50'))}, p90={fmt(a_low.get('p90'))}, p99={fmt(a_low.get('p99'))}"
        )
        print(
            "  per-case median mse(k<=k_eval): "
            f"p50={fmt(pcm.get('p50'))}, p90={fmt(pcm.get('p90'))}, "
            f"frac<1e-2={fmt(pcm.get('frac_lt_1e-2'))} (count={pcm.get('count')})"
        )

    if "error" in mp:
        print(f"[mixed_precision] {mp['error']}")
    else:
        d = mp["mse_fp16_minus_fp32"]
        fs = mp["forward_time_speedup_fp16_over_fp32"]
        ts = mp["total_time_speedup_fp16_over_fp32"]
        print(f"[mixed_precision] model={mp['model']} N={mp['n_target']} k_eval={mp['k_eval']}")
        print(
            "  mse(fp16 - fp32): "
            f"p50={fmt(d.get('p50'))}, p90={fmt(d.get('p90'))}, p99={fmt(d.get('p99'))}"
        )
        print(
            "  forward speedup(fp16 vs fp32): "
            f"p50={fmt(fs.get('p50'))}x, p90={fmt(fs.get('p90'))}x (count={fs.get('count')})"
        )
        print(
            "  total speedup(fp16 vs fp32): "
            f"p50={fmt(ts.get('p50'))}x, p90={fmt(ts.get('p90'))}x (count={ts.get('count')})"
        )


if __name__ == "__main__":
    main()
