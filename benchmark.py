import multiprocessing
import threading
import time
import subprocess
import os
import sys
import argparse
import random
import math
import queue
import json
import platform
import socket
import getpass
import numpy as np
from datetime import datetime, timedelta

def prime_factorization_worker(start_range, target_count, result_queue, completion_queue):
    """
    Worker function that performs a fixed number of prime factorizations.
    """
    def factorize(n):
        """Find all prime factors of n"""
        factors = []
        d = 2
        while d * d <= n:
            while (n % d) == 0:
                factors.append(d)
                n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    start_time = time.time()
    current = start_range
    completed = 0
    
    while completed < target_count:
        # Use larger numbers for more computational work
        target = current * 982451653 + 1000000007  # Much larger semi-primes
        factors = factorize(target)
        
        completed += 1
        current += 1
        
        # Report progress every 100 computations
        if completed % 100 == 0:
            try:
                result_queue.put(("prime_progress", completed, target_count), block=False)
            except:
                pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    try:
        completion_queue.put(("prime_complete", completed, duration), block=False)
    except:
        pass

def monte_carlo_pi_worker(target_points, result_queue, completion_queue):
    """
    Worker function that estimates Pi using a fixed number of Monte Carlo points.
    """
    start_time = time.time()
    inside_circle = 0
    total_points = 0
    
    batch_size = 1_000_000  # Report progress every 1M points
    
    while total_points < target_points:
        batch_inside = 0
        current_batch_size = min(batch_size, target_points - total_points)
        
        for _ in range(current_batch_size):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x*x + y*y <= 1:
                batch_inside += 1
        
        inside_circle += batch_inside
        total_points += current_batch_size
        
        pi_estimate = 4 * inside_circle / total_points
        try:
            result_queue.put(("pi_progress", total_points, target_points, pi_estimate), block=False)
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    final_pi = 4 * inside_circle / total_points
    
    try:
        completion_queue.put(("pi_complete", total_points, final_pi, duration), block=False)
    except:
        pass

def matrix_computation_worker(target_operations, matrix_size, result_queue, completion_queue):
    """
    Worker function that performs a fixed number of matrix operations.
    """
    start_time = time.time()
    completed = 0
    
    while completed < target_operations:
        # Generate random matrices
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        C = np.random.rand(matrix_size, matrix_size)
        
        # More intensive matrix operations
        # Matrix chain multiplication
        result1 = np.dot(A, B)
        result2 = np.dot(result1, C)
        result3 = np.dot(C, result1)
        
        try:
            # Matrix inversion
            A_inv = np.linalg.inv(A)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(A)
            
            # SVD
            U, s, Vt = np.linalg.svd(A)
            
            # QR decomposition
            Q, R = np.linalg.qr(A)
            
            # Matrix power (computationally expensive)
            A_squared = np.linalg.matrix_power(A, 2)
            
        except:
            pass  # Some matrices might be singular
        
        completed += 1
        
        # Report progress every 10 operations
        if completed % 10 == 0:
            try:
                result_queue.put(("matrix_progress", completed, target_operations), block=False)
            except:
                pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    try:
        completion_queue.put(("matrix_complete", completed, duration), block=False)
    except:
        pass

def dataset_analysis_worker(target_analyses, dataset_size, result_queue, completion_queue):
    """
    Worker function that performs a fixed number of dataset analyses.
    """
    start_time = time.time()
    completed = 0
    
    while completed < target_analyses:
        # Generate large dataset
        data = np.random.normal(100, 15, dataset_size)
        noise = np.random.normal(0, 5, dataset_size)
        combined_data = data + noise
        
        # More comprehensive statistical analysis
        mean_val = np.mean(combined_data)
        std_val = np.std(combined_data)
        median_val = np.median(combined_data)
        
        # Percentiles
        percentiles = np.percentile(combined_data, [25, 50, 75, 90, 95, 99])
        
        # Sort the data (memory intensive)
        sorted_data = np.sort(combined_data)
        
        # Histogram computation with more bins
        hist, bins = np.histogram(combined_data, bins=10000)
        
        # FFT analysis on larger chunks
        fft_size = min(262144, len(combined_data))  # Up to 256K points
        fft_result = np.fft.fft(combined_data[:fft_size])
        frequency_domain_energy = np.sum(np.abs(fft_result)**2)
        
        # Power spectral density
        psd = np.abs(fft_result)**2
        
        # Correlation analysis if we have enough data
        if len(combined_data) >= 100000:
            chunk_size = 50000
            num_chunks = len(combined_data) // chunk_size
            if num_chunks >= 2:
                chunks = combined_data[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
                correlation_matrix = np.corrcoef(chunks)
        
        completed += 1
        
        # Clean up to prevent memory issues
        del data, noise, combined_data, sorted_data, hist, fft_result, psd
        
        # Report progress every 5 analyses
        if completed % 5 == 0:
            try:
                result_queue.put(("dataset_progress", completed, target_analyses), block=False)
            except:
                pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    try:
        completion_queue.put(("dataset_complete", completed, duration), block=False)
    except:
        pass

def save_machine_info(save_path):
    """Save machine information using the machine.py script."""
    try:
        cmd = [sys.executable, "machine.py", "--save", "--path", save_path, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Machine info saved")
        return True
    except subprocess.CalledProcessError as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Error saving machine info: {e}")
        return False
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Unexpected error: {e}")
        return False

def machine_info_monitor(save_path, stop_event, interval_seconds=10):
    """Monitor function that saves machine info at specified intervals."""
    print(f"Starting machine info monitoring, saving to: {save_path}")
    
    save_machine_info(save_path)
    
    while not stop_event.is_set():
        if stop_event.wait(timeout=interval_seconds):
            break
        save_machine_info(save_path)
    
    save_machine_info(save_path)
    print("Machine info monitoring stopped")

def progress_monitor(result_queue, completion_queue, stop_event, expected_completions):
    """Monitor and display computation progress."""
    completed_tasks = 0
    task_results = {}
    
    while not stop_event.is_set() and completed_tasks < expected_completions:
        try:
            # Check for progress updates
            try:
                result = result_queue.get(timeout=1)
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if result[0] == "prime_progress":
                    _, completed, total = result
                    progress = (completed / total) * 100
                    print(f"[{timestamp}] Prime factorization: {completed:,}/{total:,} ({progress:.1f}%)")
                elif result[0] == "pi_progress":
                    _, points, total, pi_est = result
                    progress = (points / total) * 100
                    error = abs(pi_est - math.pi)
                    print(f"[{timestamp}] Monte Carlo Pi: {points:,}/{total:,} ({progress:.1f}%), π≈{pi_est:.6f} (error: {error:.6f})")
                elif result[0] == "matrix_progress":
                    _, completed, total = result
                    progress = (completed / total) * 100
                    print(f"[{timestamp}] Matrix operations: {completed:,}/{total:,} ({progress:.1f}%)")
                elif result[0] == "dataset_progress":
                    _, completed, total = result
                    progress = (completed / total) * 100
                    print(f"[{timestamp}] Dataset analysis: {completed:,}/{total:,} ({progress:.1f}%)")
                    
            except queue.Empty:
                pass
            
            # Check for completions
            try:
                completion = completion_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if completion[0] == "prime_complete":
                    _, count, duration = completion
                    rate = count / duration
                    print(f"[{timestamp}] ✓ Prime factorization complete: {count:,} in {duration:.2f}s ({rate:.1f}/sec)")
                    task_results["prime_factorization"] = {"count": count, "duration": duration, "rate": rate}
                    completed_tasks += 1
                elif completion[0] == "pi_complete":
                    _, points, pi_est, duration = completion
                    error = abs(pi_est - math.pi)
                    rate = points / duration
                    print(f"[{timestamp}] ✓ Monte Carlo Pi complete: π={pi_est:.8f} (error: {error:.8f}) in {duration:.2f}s ({rate:,.0f} points/sec)")
                    task_results["monte_carlo_pi"] = {"points": points, "pi_estimate": pi_est, "error": error, "duration": duration, "rate": rate}
                    completed_tasks += 1
                elif completion[0] == "matrix_complete":
                    _, count, duration = completion
                    rate = count / duration
                    print(f"[{timestamp}] ✓ Matrix operations complete: {count:,} in {duration:.2f}s ({rate:.2f}/sec)")
                    task_results["matrix_operations"] = {"count": count, "duration": duration, "rate": rate}
                    completed_tasks += 1
                elif completion[0] == "dataset_complete":
                    _, count, duration = completion
                    rate = count / duration
                    print(f"[{timestamp}] ✓ Dataset analysis complete: {count:,} in {duration:.2f}s ({rate:.2f}/sec)")
                    task_results["dataset_analysis"] = {"count": count, "duration": duration, "rate": rate}
                    completed_tasks += 1
                    
            except queue.Empty:
                pass
                
        except Exception as e:
            print(f"Error in progress monitoring: {e}")
            continue
    
    print("\n" + "="*50)
    print("ALL COMPUTATIONS COMPLETED!")
    print("="*50)
    
    return task_results

def save_benchmark_results(save_path, workload_size, workload, total_duration, cpu_count, task_results, saved_files_count):
    """Save benchmark results to a JSON file."""
    
    # Get system information
    system_info = {
        "hostname": socket.gethostname(),
        "username": getpass.getuser(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": cpu_count
    }
    
    # Create comprehensive results
    benchmark_results = {
        "benchmark_info": {
            "timestamp": datetime.now().isoformat(),
            "workload_size": workload_size,
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_minutes": round(total_duration / 60, 2),
            "total_duration_hours": round(total_duration / 3600, 2) if total_duration >= 3600 else None,
            "performance_score": round(10000 / total_duration, 2),
            "machine_info_snapshots": saved_files_count
        },
        "system_info": system_info,
        "workload_specification": workload,
        "task_results": task_results,
        "performance_metrics": {
            "total_prime_factorizations": sum(result.get("count", 0) for result in task_results.values() if "count" in result and "prime" in str(result)),
            "total_monte_carlo_points": task_results.get("monte_carlo_pi", {}).get("points", 0),
            "pi_estimation_error": task_results.get("monte_carlo_pi", {}).get("error", None),
            "total_matrix_operations": task_results.get("matrix_operations", {}).get("count", 0),
            "total_dataset_analyses": task_results.get("dataset_analysis", {}).get("count", 0)
        }
    }
    
    # Save to JSON file
    results_file = os.path.join(save_path, "benchmark_results.json")
    try:
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=4)
        print(f"📊 Benchmark results saved to: {results_file}")
        return results_file
    except Exception as e:
        print(f"Error saving benchmark results: {e}")
        return None

def run_heavy_workload_benchmark(save_path="heavy_benchmark_results", workload_size="medium"):
    """
    Run computational benchmark with much heavier workload sizes.
    """
    # Define much larger workload sizes
    workloads = {
        "small": {
            "prime_factorizations": 5_000,           # 10x increase
            "monte_carlo_points": 500_000_000,       # 50x increase (500M)
            "matrix_operations": 500,                # 10x increase  
            "matrix_size": 200,                      # Larger matrices
            "dataset_analyses": 100,                 # 5x increase
            "dataset_size": 2_000_000               # 4x increase (2M)
        },
        "medium": {
            "prime_factorizations": 20_000,          # 10x increase
            "monte_carlo_points": 2_000_000_000,     # 40x increase (2B)
            "matrix_operations": 1_000,              # 10x increase
            "matrix_size": 300,                      # Much larger matrices
            "dataset_analyses": 200,                 # 4x increase
            "dataset_size": 5_000_000               # 5x increase (5M)
        },
        "large": {
            "prime_factorizations": 50_000,          # 10x increase
            "monte_carlo_points": 10_000_000_000,    # 50x increase (10B)
            "matrix_operations": 2_000,              # 10x increase
            "matrix_size": 400,                      # Much larger matrices
            "dataset_analyses": 500,                 # 5x increase
            "dataset_size": 10_000_000              # 5x increase (10M)
        },
        "extreme": {
            "prime_factorizations": 100_000,         # New tier
            "monte_carlo_points": 50_000_000_000,    # 50B points
            "matrix_operations": 5_000,              # Massive matrix work
            "matrix_size": 500,                      # Very large matrices
            "dataset_analyses": 1_000,               # Extensive analysis
            "dataset_size": 20_000_000              # 20M points per analysis
        }
    }
    
    if workload_size not in workloads:
        raise ValueError(f"Unknown workload size: {workload_size}")
    
    workload = workloads[workload_size]
    cpu_count = multiprocessing.cpu_count()
    
    print("=" * 70)
    print("HEAVY COMPUTATIONAL BENCHMARK")
    print("=" * 70)
    print(f"Workload size: {workload_size}")
    print(f"CPU cores available: {cpu_count}")
    print(f"Save path: {save_path}")
    print()
    print("Workload breakdown:")
    for task, amount in workload.items():
        if isinstance(amount, int):
            print(f"  - {task}: {amount:,}")
        else:
            print(f"  - {task}: {amount}")
    print("-" * 70)
    
    # Estimate time based on workload
    time_estimates = {
        "small": "5-15 minutes",
        "medium": "15-45 minutes", 
        "large": "45-120 minutes",
        "extreme": "2-6 hours"
    }
    print(f"Estimated completion time: {time_estimates.get(workload_size, 'Unknown')}")
    print("-" * 70)
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Start timing
    benchmark_start = time.time()
    
    # Create queues for communication
    result_queue = multiprocessing.Queue()
    completion_queue = multiprocessing.Queue()
    stop_event = threading.Event()
    
    # Start machine info monitoring (more frequent for longer runs)
    monitor_thread = threading.Thread(
        target=machine_info_monitor,
        args=(save_path, stop_event, 10),  # Save every 10 seconds
        daemon=True
    )
    monitor_thread.start()
    
    # Calculate process distribution
    processes_per_type = max(1, cpu_count // 4)
    expected_completions = processes_per_type * 4
    
    # Start progress monitoring
    progress_thread = threading.Thread(
        target=progress_monitor,
        args=(result_queue, completion_queue, stop_event, expected_completions),
        daemon=True
    )
    progress_thread.start()
    
    # Start computation processes
    processes = []
    
    # Prime factorization processes
    factors_per_process = workload["prime_factorizations"] // processes_per_type
    for i in range(processes_per_type):
        start_range = 1000000 + i * 1000000
        process = multiprocessing.Process(
            target=prime_factorization_worker,
            args=(start_range, factors_per_process, result_queue, completion_queue)
        )
        process.start()
        processes.append(process)
    
    # Monte Carlo Pi processes  
    points_per_process = workload["monte_carlo_points"] // processes_per_type
    for i in range(processes_per_type):
        process = multiprocessing.Process(
            target=monte_carlo_pi_worker,
            args=(points_per_process, result_queue, completion_queue)
        )
        process.start()
        processes.append(process)
    
    # Matrix computation processes
    ops_per_process = workload["matrix_operations"] // processes_per_type
    for i in range(processes_per_type):
        matrix_size = workload["matrix_size"] + i * 50  # Vary matrix sizes more
        process = multiprocessing.Process(
            target=matrix_computation_worker,
            args=(ops_per_process, matrix_size, result_queue, completion_queue)
        )
        process.start()
        processes.append(process)
    
    # Dataset analysis processes
    analyses_per_process = workload["dataset_analyses"] // processes_per_type
    for i in range(processes_per_type):
        process = multiprocessing.Process(
            target=dataset_analysis_worker,
            args=(analyses_per_process, workload["dataset_size"], result_queue, completion_queue)
        )
        process.start()
        processes.append(process)
    
    print(f"Started {len(processes)} computation processes...")
    print(f"Each type using {processes_per_type} processes")
    print()
    
    # Store task results
    task_results = {}
    
    try:
        # Wait for all processes to complete
        for process in processes:
            process.join()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user!")
    
    finally:
        # Stop monitoring and get final task results
        stop_event.set()
        monitor_thread.join(timeout=2)
        
        # Get final results from progress monitor
        try:
            # Give progress monitor a moment to collect final results
            time.sleep(1)
        except:
            pass
        
        progress_thread.join(timeout=2)
        
        # Terminate any remaining processes
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
    
    # Calculate total time
    benchmark_end = time.time()
    total_duration = benchmark_end - benchmark_start
    
    print("\n" + "="*70)
    print("HEAVY BENCHMARK RESULTS")
    print("="*70)
    print(f"Workload size: {workload_size}")
    print(f"Total runtime: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    if total_duration >= 3600:
        print(f"               {total_duration/3600:.2f} hours")
    print(f"CPU cores used: {cpu_count}")
    print(f"Performance score: {10000/total_duration:.2f} heavy-points/second")
    
    # Count saved files
    saved_files_count = 0
    try:
        saved_files = [f for f in os.listdir(save_path) if f.endswith('.json')]
        saved_files_count = len(saved_files)
        print(f"Machine info snapshots saved: {saved_files_count}")
        print(f"Files location: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"Error checking saved files: {e}")
    
    # Save comprehensive benchmark results to JSON
    results_file = save_benchmark_results(
        save_path, workload_size, workload, total_duration, 
        cpu_count, task_results, saved_files_count
    )
    
    print("="*70)
    return total_duration, results_file

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Heavy Computational Benchmark for Powerful Machines with Results Saving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs much heavier computational workloads designed for powerful machines
and saves comprehensive results to JSON files for easy comparison between machines.

Choose workload size based on your time constraints:

- small:   ~5-15 minutes on powerful machines
- medium:  ~15-45 minutes on powerful machines  
- large:   ~45-120 minutes on powerful machines
- extreme: ~2-6 hours on powerful machines

Examples:
  python benchmark_with_results.py                                    # Medium workload
  python benchmark_with_results.py --workload small                  # Quick benchmark
  python benchmark_with_results.py --workload large --path /tmp/bench # Long benchmark
  python benchmark_with_results.py --workload extreme                # Maximum challenge
  python benchmark_with_results.py --path ./results_${SLURM_JOB_NAME} # For SLURM jobs
        """
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default="heavy_benchmark_results",
        help='Base path where to save machine info JSON files (default: heavy_benchmark_results)'
    )
    
    parser.add_argument(
        '--workload', '-w',
        type=str,
        choices=['small', 'medium', 'large', 'extreme'],
        default='medium',
        help='Workload size: small, medium, large, extreme (default: medium)'
    )
    
    args = parser.parse_args()
    
    # Check if machine.py exists
    if not os.path.exists("machine.py"):
        print("Error: machine.py not found in current directory!")
        print("Please run this script from the same directory as machine.py")
        return 1
    
    # Run the benchmark
    try:
        duration, results_file = run_heavy_workload_benchmark(
            save_path=args.path,
            workload_size=args.workload
        )
        
        print(f"\n🏁 Heavy benchmark completed in {duration:.2f} seconds!")
        if duration >= 3600:
            print(f"   That's {duration/3600:.2f} hours of intense computation!")
        elif duration >= 60:
            print(f"   That's {duration/60:.2f} minutes of intense computation!")
        print(f"💾 Results saved to: {os.path.abspath(args.path)}")
        if results_file:
            print(f"📊 Summary results: {results_file}")
        return 0
        
    except Exception as e:
        print(f"Error running heavy benchmark: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
