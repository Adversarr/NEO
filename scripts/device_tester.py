import torch
import time
import argparse
import sys
import os
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# CUDA Kernel for Pointer Chasing (Latency Test)
# -----------------------------------------------------------------------------
# This is a classic pointer-chasing algorithm that forces the GPU to access memory
# serially, bypassing the L1/L2 cache, to measure the true Global Memory Latency.
pointer_chasing_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void pointer_chasing_kernel(int64_t* data, int64_t* output, int iterations) {
    int64_t p = 0;
    int64_t start = clock64();
    // Force serial jumps: each access depends on the result of the previous one
    for (int i = 0; i < iterations; i++) {
        p = data[p]; 
    }
    int64_t end = clock64();
    
    output[0] = (end - start);
    output[1] = p; // Write a dummy value to prevent the compiler from optimizing away the loop
}

std::vector<int64_t> run_latency_test(torch::Tensor data, int iterations) {
    auto output = torch::zeros({2}, torch::dtype(torch::kInt64).device(data.device()));
    
    // Launch kernel with 1 block, 1 thread
    pointer_chasing_kernel<<<1, 1>>>(data.data_ptr<int64_t>(), output.data_ptr<int64_t>(), iterations);
    
    return {output[0].item<int64_t>(), output[1].item<int64_t>()};
}
"""

pointer_chasing_cpp_source = """
std::vector<int64_t> run_latency_test(torch::Tensor data, int iterations);
"""

# -----------------------------------------------------------------------------
# Benchmark Utilities
# -----------------------------------------------------------------------------

def print_header(title):
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}")

def get_device_info(device_id):
    props = torch.cuda.get_device_properties(device_id)
    print_header("GPU Device Info")
    print(f"Device Name:       {props.name}")
    print(f"Compute Cap:       {props.major}.{props.minor}")
    print(f"Total VRAM:        {props.total_memory / 1024**3:.2f} GB")
    print(f"MultiProcessors:   {props.multi_processor_count}")
    try:
        mem_clock_rate = getattr(props, "memory_clock_rate", 0)
        mem_bus_width = getattr(props, "memory_bus_width", 0)
        if mem_clock_rate > 0 and mem_bus_width > 0:
            mem_clock_hz = float(mem_clock_rate) * 1e3
            bytes_per_cycle = (mem_bus_width / 8.0)
            theoretical_bw = 2.0 * mem_clock_hz * bytes_per_cycle / 1e9
            print(f"Mem Bus Width:     {mem_bus_width} bits")
            print(f"Mem Clock:         {mem_clock_rate / 1000.0:.2f} MHz (raw)")
            print(f"Theoretical BW:    {theoretical_bw:.2f} GB/s (GDDR, peak)")
    except Exception:
        pass
    print(f"CUDA Version:      {torch.version.cuda}")
    return props

def benchmark_gemm(device, dtype, sizes, num_iters=20, warmup=5):
    dtype_str = "FP32" if dtype == torch.float32 else "BF16"
    print_header(f"{dtype_str} GEMM Compute Benchmark")

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Warning: BF16 is not supported on this device, skipping.")
        return

    print(f"{'Matrix Size (NxN)':<20} | {'Time (ms)':<15} | {'TFLOPS':<15}")
    print("-" * 56)

    for n in sizes:
        try:
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            
            for _ in range(warmup):
                torch.matmul(a, b)
            torch.cuda.synchronize()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_iters):
                torch.matmul(a, b)
            end_event.record()
            torch.cuda.synchronize()

            elapsed_time_ms = start_event.elapsed_time(end_event) / num_iters
            
            flops = 2 * (n ** 3)
            tflops = (flops / (elapsed_time_ms / 1000.0)) / 1e12
            
            print(f"{n:<20} | {elapsed_time_ms:.4f} ms      | {tflops:.4f}")
            
            del a, b
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{n:<20} | OOM (out of VRAM)  | -")
                torch.cuda.empty_cache()
            else:
                print(f"Error at size {n}: {e}")

def benchmark_memory_bandwidth(device, size_gb, num_iters=20, warmup=5):
    print_header("VRAM Bandwidth Benchmark")
    
    numel = int(size_gb * 1024**3 // 4)
    try:
        t1 = torch.empty(numel, device=device, dtype=torch.float32).normal_()
        t2 = torch.empty_like(t1)
    except RuntimeError:
        print(f"Cannot allocate {size_gb}GB of VRAM, trying to reduce test size automatically...")
        try:
            size_gb = 1.0
            numel = int(size_gb * 1024**3 // 4)
            t1 = torch.empty(numel, device=device, dtype=torch.float32).normal_()
            t2 = torch.empty_like(t1)
            print(f"Reduced to testing a {size_gb}GB block.")
        except:
             print("Insufficient VRAM, skipping bandwidth test.")
             return

    for _ in range(warmup):
        t2.copy_(t1)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        t2.copy_(t1) # D2D copy
    end_event.record()
    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / num_iters
    total_bytes_moved = (numel * 4) * 2 
    gb_moved = total_bytes_moved / 1024**3
    bw = gb_moved / (avg_ms / 1000.0)

    print(f"Test block size:   {size_gb:.2f} GB")
    print(f"Avg time:          {avg_ms:.4f} ms")
    print(f"Measured BW:       {bw:.2f} GB/s")
    print("Note: this value should be close to the theoretical peak bandwidth of the GPU (e.g. RTX 3090 ≈ 936 GB/s)")

def benchmark_pcie_bandwidth(device, size_mb=512, num_iters=50):
    print_header("PCIe Bus Bandwidth Benchmark")
    
    size_bytes = size_mb * 1024 * 1024
    cpu_tensor = torch.randn(size_bytes // 4, dtype=torch.float32).pin_memory()
    gpu_tensor = torch.empty(size_bytes // 4, dtype=torch.float32, device=device)
    
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    h2d_time = (time.time() - start) / num_iters
    h2d_bw = (size_bytes / 1024**3) / h2d_time
    
    start = time.time()
    for _ in range(num_iters):
        cpu_tensor.copy_(gpu_tensor, non_blocking=True)
    torch.cuda.synchronize()
    d2h_time = (time.time() - start) / num_iters
    d2h_bw = (size_bytes / 1024**3) / d2h_time
    
    print(f"Host to Device (CPU->GPU): {h2d_bw:.2f} GB/s")
    print(f"Device to Host (GPU->CPU): {d2h_bw:.2f} GB/s")
    
    if h2d_bw < 6.0:
        print("\n[Warning] PCIe bandwidth is too low! Expected: PCIe 3.0 x16 > 11 GB/s, PCIe 4.0 x16 > 24 GB/s.")
        print("Possible causes: riser cable, x4 slot, or GPU power-saving mode.")

def benchmark_latency(device, size_mb=256, iterations=50000):
    print_header("VRAM Physical Latency Benchmark (Global Memory Latency)")
    print("Compiling CUDA Kernel...")
    
    try:
        latency_module = load_inline(
            name='latency_test',
            cpp_sources=pointer_chasing_cpp_source,
            cuda_sources=pointer_chasing_source,
            functions=['run_latency_test'],
            with_cuda=True,
            extra_cuda_cflags=["-O3"]
        )
    except Exception as e:
        print("\n[Error] Failed to compile CUDA code.")
        print("Likely cause: full CUDA Toolkit not installed (only PyTorch runtime present), or nvcc not found.")
        print("Cannot perform nanosecond-level latency test. Skipping.")
        return

    num_elements = size_mb * 1024 * 1024 // 8 # int64
    stride = 1024 * 1024 # Jump stride to create cache misses
    
    index_array = torch.arange(num_elements, dtype=torch.int64, device=device)
    perm = (index_array + stride) % num_elements
    perm = perm.contiguous()
    
    latency_module.run_latency_test(perm, 1000)
    torch.cuda.synchronize()
    
    ret = latency_module.run_latency_test(perm, iterations)
    total_clocks = ret[0]
    
    avg_clocks = total_clocks / iterations
    
    print(f"Test region size:  {size_mb} MB")
    print(f"Jump stride:       {stride * 8 / 1024:.2f} KB")
    print(f"Avg access latency:{avg_clocks:.2f} Cycles (clock cycles)")
    print("\nReference values:")
    print("Ampere (30xx/40xx) typical: 400 ~ 600 Cycles (depending on L2 hit rate)")
    print("Above 1000 Cycles indicates very slow VRAM response or very low hit rate.")
    print("Note: if the GPU is in a low-power idle state, cycle counts will be inflated. Ensure high-performance mode.")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU Deep Benchmark Tool (PyTorch/CUDA)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--fp32", action="store_true", help="Benchmark FP32 GEMM")
    parser.add_argument("--bf16", action="store_true", help="Benchmark BF16 GEMM")
    parser.add_argument("--bw", action="store_true", help="Benchmark VRAM bandwidth")
    parser.add_argument("--latency", action="store_true", help="Benchmark VRAM latency")
    parser.add_argument("--pcie", action="store_true", help="Benchmark PCIe bandwidth")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks at once")
    parser.add_argument("--gemm-sizes", type=str, default=None, help="Custom GEMM matrix sizes, e.g.: 2048,4096,8192")
    parser.add_argument("--gemm-iters", type=int, default=20, help="Number of GEMM timing iterations per matrix size")
    parser.add_argument("--gemm-warmup", type=int, default=5, help="Number of GEMM warmup iterations")
    parser.add_argument("--bw-size-gb", type=float, default=None, help="VRAM bandwidth test block size (GB); default is 80% of free VRAM, capped at 4GB")
    parser.add_argument("--bw-iters", type=int, default=20, help="Number of VRAM bandwidth test iterations")
    parser.add_argument("--bw-warmup", type=int, default=5, help="Number of VRAM bandwidth warmup iterations")
    parser.add_argument("--pcie-size-mb", type=int, default=512, help="PCIe bandwidth test transfer size per iteration (MB)")
    parser.add_argument("--pcie-iters", type=int, default=50, help="Number of PCIe bandwidth test iterations")
    parser.add_argument("--latency-region-mb", type=int, default=256, help="Address space size covered by VRAM latency test (MB)")
    parser.add_argument("--latency-iters", type=int, default=50000, help="Number of pointer-chasing jumps (more = more stable but slower)")
    
    args = parser.parse_args()

    if args.all:
        args.fp32 = True
        args.bf16 = True
        args.bw = True
        args.latency = True
        args.pcie = True

    if not any([args.fp32, args.bf16, args.bw, args.latency, args.pcie]):
        print("No benchmark explicitly selected; running all by default (--all).")
        args.fp32 = True
        args.bf16 = True
        args.bw = True
        args.latency = True
        args.pcie = True

    if not torch.cuda.is_available():
        print("Error: No CUDA device detected. Please check your driver installation.")
        sys.exit(1)

    device = torch.device(f"cuda:{args.device}")
    
    props = get_device_info(args.device)
    
    # Enable cuDNN auto-tuning for best GEMM performance
    torch.backends.cudnn.benchmark = True
    
    if args.pcie:
        benchmark_pcie_bandwidth(device, size_mb=args.pcie_size_mb, num_iters=args.pcie_iters)
            
    if args.bw:
        if args.bw_size_gb is not None:
            test_size = float(args.bw_size_gb)
        else:
            free_mem, total_mem = torch.cuda.mem_get_info(args.device)
            test_size = min(4.0, (free_mem * 0.8) / 1024**3)
        benchmark_memory_bandwidth(device, test_size, num_iters=args.bw_iters, warmup=args.bw_warmup)

    if args.fp32:
        mem_gb = props.total_memory / 1024**3
        if mem_gb > 16:
            sizes = [4096, 8192, 16384] 
        elif mem_gb > 8:
            sizes = [2048, 4096, 8192]
        else:
            sizes = [1024, 2048, 4096]
        if args.gemm_sizes is not None:
            user_sizes = []
            for part in args.gemm_sizes.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    user_sizes.append(int(part))
                except ValueError:
                    continue
            if user_sizes:
                sizes = user_sizes
        benchmark_gemm(device, torch.float32, sizes, num_iters=args.gemm_iters, warmup=args.gemm_warmup)

    if args.bf16:
        mem_gb = props.total_memory / 1024**3
        if mem_gb > 16:
            sizes = [4096, 8192, 16384]
        else:
            sizes = [2048, 4096, 8192]
        if args.gemm_sizes is not None:
            user_sizes = []
            for part in args.gemm_sizes.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    user_sizes.append(int(part))
                except ValueError:
                    continue
            if user_sizes:
                sizes = user_sizes
        benchmark_gemm(device, torch.bfloat16, sizes, num_iters=args.gemm_iters, warmup=args.gemm_warmup)

    if args.latency:
        benchmark_latency(device, size_mb=args.latency_region_mb, iterations=args.latency_iters)

    print_header("Benchmarks Complete")

if __name__ == "__main__":
    main()
