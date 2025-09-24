#!/usr/bin/env python3
"""
Computational Heavy Problem Solver with Machine Info Monitoring

This script solves computationally intensive problems (prime factorization,
Monte Carlo Pi estimation, matrix operations) while monitoring machine
performance using machine.py.
"""

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
import numpy as np
from datetime import datetime, timedelta

def prime_factorization_worker(start_range, result_queue):
    """
    Worker function that performs prime factorization on large numbers.
    
    Args:
        start_range (int): Starting number for the range to factorize
        result_queue (multiprocessing.Queue): Queue to store results
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
    
    results = {}
    current = start_range
    
    while True:
        # Work on increasingly large numbers for factorization
        if current > 10**12:  # Reset to avoid infinite growth
            current = start_range
        
        # Find a large number to factorize (semi-prime for difficulty)
        target = current * 1009 + 982451653  # Large semi-prime generation
        factors = factorize(target)
        
        results[target] = factors
        
        # Store result every 100 computations
        if len(results) >= 100:
            try:
                result_queue.put(("prime_factors", len(results)), block=False)
            except:
                pass
            results.clear()
        
        current += 1

def monte_carlo_pi_worker(iterations_per_batch, result_queue):
    """
    Worker function that estimates Pi using Monte Carlo method.
    
    Args:
        iterations_per_batch (int): Number of iterations per batch
        result_queue (multiprocessing.Queue): Queue to store results
    """
    inside_circle = 0
    total_points = 0
    
    while True:
        # Generate random points and check if they're inside unit circle
        for _ in range(iterations_per_batch):
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            if x*x + y*y <= 1:
                inside_circle += 1
            total_points += 1
        
        # Estimate Pi
        pi_estimate = 4 * inside_circle / total_points
        
        try:
            result_queue.put(("pi_estimate", pi_estimate, total_points), block=False)
        except:
            pass

def matrix_computation_worker(matrix_size, result_queue):
    """
    Worker function that performs heavy matrix computations.
    
    Args:
        matrix_size (int): Size of matrices to compute
        result_queue (multiprocessing.Queue): Queue to store results
    """
    computation_count = 0
    
    while True:
        # Generate random matrices
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)
        
        # Perform heavy matrix operations
        # Matrix multiplication
        C = np.dot(A, B)
        
        # Matrix inversion (if possible)
        try:
            A_inv = np.linalg.inv(A)
            # Verify inversion
            identity_check = np.allclose(np.dot(A, A_inv), np.eye(matrix_size))
        except:
            identity_check = False
        
        # Eigenvalue decomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            max_eigenvalue = np.max(np.real(eigenvalues))
        except:
            max_eigenvalue = 0
        
        # Singular Value Decomposition
        try:
            U, s, Vt = np.linalg.svd(A)
            condition_number = np.max(s) / np.min(s)
        except:
            condition_number = float('inf')
        
        computation_count += 1
        
        # Report results every 10 computations
        if computation_count % 10 == 0:
            try:
                result_queue.put((
                    "matrix_ops", 
                    computation_count, 
                    max_eigenvalue, 
                    condition_number,
                    identity_check
                ), block=False)
            except:
                pass

def large_dataset_analysis_worker(result_queue):
    """
    Worker function that performs analysis on large datasets.
    Creates memory pressure while solving meaningful problems.
    """
    analysis_count = 0
    
    while True:
        try:
            # Generate large dataset (uses significant memory)
            dataset_size = 1000000  # 1 million data points
            data = np.random.normal(100, 15, dataset_size)  # Normal distribution
            
            # Statistical analysis
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            
            # Create additional memory pressure with complex operations
            # Sort the data (memory intensive)
            sorted_data = np.sort(data)
            
            # Histogram computation
            hist, bins = np.histogram(data, bins=1000)
            
            # Correlation matrix (if treating as time series chunks)
            chunk_size = 10000
            chunks = data[:dataset_size//chunk_size * chunk_size].reshape(-1, chunk_size)
            correlation_matrix = np.corrcoef(chunks)
            
            # FFT analysis (computationally and memory intensive)
            fft_result = np.fft.fft(data[:65536])  # Use power of 2 for efficiency
            frequency_domain_energy = np.sum(np.abs(fft_result)**2)
            
            analysis_count += 1
            
            # Report results
            try:
                result_queue.put((
                    "dataset_analysis",
                    analysis_count,
                    mean_val,
                    std_val,
                    median_val,
                    frequency_domain_energy,
                    correlation_matrix.shape
                ), block=False)
            except:
                pass
                
            # Clean up large arrays to prevent excessive memory growth
            del data, sorted_data, hist, chunks, correlation_matrix, fft_result
            
        except MemoryError:
            print("Warning: Memory limit reached in dataset analysis")
            time.sleep(1)
        except Exception as e:
            print(f"Error in dataset analysis: {e}")
            time.sleep(1)

def result_monitor(result_queue, stop_event):
    """
    Monitor and display computation results from worker processes.
    
    Args:
        result_queue (multiprocessing.Queue): Queue containing computation results
        stop_event (threading.Event): Event to signal when to stop monitoring
    """
    pi_estimates = []
    prime_factor_count = 0
    matrix_computation_count = 0
    dataset_analysis_count = 0
    
    print("Starting computation result monitoring...")
    
    while not stop_event.is_set():
        try:
            # Check for results with timeout
            result = result_queue.get(timeout=2)
            
            if result[0] == "pi_estimate":
                _, pi_est, total_points = result
                pi_estimates.append(pi_est)
                if len(pi_estimates) % 10 == 0:  # Report every 10 estimates
                    avg_pi = sum(pi_estimates[-10:]) / min(10, len(pi_estimates))
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] Monte Carlo Pi estimate: {avg_pi:.6f} (from {total_points} points)")
                    
            elif result[0] == "prime_factors":
                _, count = result
                prime_factor_count += count
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Prime factorizations completed: {prime_factor_count}")
                
            elif result[0] == "matrix_ops":
                _, count, max_eigen, cond_num, identity_ok = result
                matrix_computation_count = count
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Matrix operations: {count}, Max eigenvalue: {max_eigen:.3f}, Condition: {cond_num:.2e}")
                
            elif result[0] == "dataset_analysis":
                _, count, mean_val, std_val, median_val, freq_energy, corr_shape = result
                dataset_analysis_count = count
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] Dataset analysis #{count}: μ={mean_val:.2f}, σ={std_val:.2f}, correlation matrix: {corr_shape}")
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Error in result monitoring: {e}")
            continue
    
    # Final summary
    print("\n" + "="*50)
    print("COMPUTATION SUMMARY")
    print("="*50)
    if pi_estimates:
        final_pi = sum(pi_estimates) / len(pi_estimates)
        print(f"Final Pi estimate: {final_pi:.8f} (error: {abs(final_pi - math.pi):.8f})")
    print(f"Prime factorizations completed: {prime_factor_count}")
    print(f"Matrix computations completed: {matrix_computation_count}")
    print(f"Dataset analyses completed: {dataset_analysis_count}")
    print("="*50)

def save_machine_info(save_path):
    """
    Save machine information using the machine.py script.
    
    Args:
        save_path (str): Path where to save the machine info
    """
    try:
        # Run the machine.py script with save and path arguments
        cmd = [sys.executable, "machine.py", "--save", "--path", save_path, "--quiet"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Machine info saved successfully")
        return True
    except subprocess.CalledProcessError as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Error saving machine info: {e}")
        print(f"[{timestamp}] STDERR: {e.stderr}")
        return False
    except Exception as e:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Unexpected error: {e}")
        return False

def machine_info_monitor(save_path, stop_event, interval_seconds=10):
    """
    Monitor function that saves machine info at specified intervals.
    
    Args:
        save_path (str): Path where to save machine info files
        stop_event (threading.Event): Event to signal when to stop monitoring
        interval_seconds (int): Interval between saves in seconds
    """
    print(f"Starting machine info monitoring, saving to: {save_path}")
    print(f"Save interval: {interval_seconds} seconds")
    
    # Save initial state
    save_machine_info(save_path)
    
    while not stop_event.is_set():
        # Wait for specified interval or until stop event
        if stop_event.wait(timeout=interval_seconds):
            break
        
        # Save machine info
        save_machine_info(save_path)
    
    # Save final state
    save_machine_info(save_path)
    print("Machine info monitoring stopped")

def run_computational_benchmark(duration_seconds=60, save_path="computation_test", cpu_processes=None, memory_processes=2, info_interval=10):
    """
    Run computationally heavy problems with machine monitoring.
    
    Args:
        duration_seconds (int): Duration of the computation in seconds
        save_path (str): Path where to save machine info files
        cpu_processes (int, optional): Number of computation processes. If None, uses CPU count
        memory_processes (int): Number of memory-intensive processes
        info_interval (int): Interval between machine info saves in seconds
    """
    print(f"Starting {duration_seconds}-second computational benchmark...")
    print(f"Machine info will be saved to: {save_path}")
    print(f"Info save interval: {info_interval} seconds")
    
    # Determine number of CPU processes
    cpu_count = multiprocessing.cpu_count()
    if cpu_processes is None:
        cpu_processes = cpu_count
    
    print(f"CPU cores available: {cpu_count}")
    print(f"Computation processes: {cpu_processes}")
    print(f"Memory-intensive processes: {memory_processes}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate end time
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    print(f"Start time: {start_time.strftime('%H:%M:%S')}")
    print(f"End time: {end_time.strftime('%H:%M:%S')}")
    print("-" * 50)
    
    # Create shared result queue for communication between processes
    result_queue = multiprocessing.Queue()
    
    # Create stop event for threads
    stop_event = threading.Event()
    
    # Start machine info monitoring thread
    monitor_thread = threading.Thread(
        target=machine_info_monitor, 
        args=(save_path, stop_event, info_interval),
        daemon=True
    )
    monitor_thread.start()
    
    # Start computation result monitoring thread
    result_monitor_thread = threading.Thread(
        target=result_monitor,
        args=(result_queue, stop_event),
        daemon=True
    )
    result_monitor_thread.start()
    
    # Create computational processes
    computation_processes = []
    
    # Distribute different types of computations across available cores
    processes_per_type = max(1, cpu_processes // 3)  # Divide among 3 computation types
    
    print(f"Starting prime factorization processes ({processes_per_type})...")
    for i in range(processes_per_type):
        start_range = 1000000 + i * 1000000  # Different starting ranges
        process = multiprocessing.Process(
            target=prime_factorization_worker, 
            args=(start_range, result_queue)
        )
        process.start()
        computation_processes.append(process)
    
    print(f"Starting Monte Carlo Pi estimation processes ({processes_per_type})...")
    for i in range(processes_per_type):
        iterations_per_batch = 100000  # 100k iterations per batch
        process = multiprocessing.Process(
            target=monte_carlo_pi_worker,
            args=(iterations_per_batch, result_queue)
        )
        process.start()
        computation_processes.append(process)
    
    print(f"Starting matrix computation processes ({cpu_processes - 2 * processes_per_type})...")
    remaining_processes = cpu_processes - 2 * processes_per_type
    for i in range(max(1, remaining_processes)):
        matrix_size = 100 + i * 50  # Different matrix sizes for variety
        process = multiprocessing.Process(
            target=matrix_computation_worker,
            args=(matrix_size, result_queue)
        )
        process.start()
        computation_processes.append(process)
    
    # Create memory-intensive processes
    memory_process_list = []
    if memory_processes > 0:
        print(f"Starting dataset analysis processes ({memory_processes})...")
        for i in range(memory_processes):
            process = multiprocessing.Process(
                target=large_dataset_analysis_worker,
                args=(result_queue,)
            )
            process.start()
            memory_process_list.append(process)
    
    try:
        # Run for the specified duration
        while datetime.now() < end_time:
            remaining = (end_time - datetime.now()).total_seconds()
            print(f"Computational benchmark running... {remaining:.1f} seconds remaining")
            time.sleep(5)  # Update every 5 seconds
        
    except KeyboardInterrupt:
        print("\nInterrupted by user!")
    
    finally:
        print("\nStopping computational benchmark...")
        
        # Stop monitoring
        stop_event.set()
        monitor_thread.join(timeout=3)
        result_monitor_thread.join(timeout=3)
        
        # Terminate all computation processes
        print("Terminating computation processes...")
        for process in computation_processes:
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
        
        print("Terminating memory-intensive processes...")
        for process in memory_process_list:
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
        
        print("Computational benchmark completed!")
        
        # Show summary
        end_actual = datetime.now()
        actual_duration = (end_actual - start_time).total_seconds()
        print(f"Actual duration: {actual_duration:.1f} seconds")
        
        # Count saved files
        try:
            saved_files = [f for f in os.listdir(save_path) if f.endswith('.json')]
            print(f"Machine info files saved: {len(saved_files)}")
            print(f"Files location: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"Error checking saved files: {e}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Computational Heavy Problem Solver with Machine Info Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script solves computationally intensive problems while monitoring machine performance:
- Prime factorization of large numbers
- Monte Carlo Pi estimation
- Matrix computations (multiplication, inversion, eigenvalues, SVD)
- Large dataset statistical analysis with FFT

Examples:
  python stress_test.py                                    # Default: 60s benchmark, save to 'computation_test/'
  python stress_test.py --path /tmp/benchmark_logs        # Save to specific directory
  python stress_test.py --duration 30 --path ./test_data  # 30s benchmark, save to relative directory
  python stress_test.py -d 120 -p /var/log/machine_bench  # 2-minute benchmark, custom path
  python stress_test.py --cpu-processes 8 --memory-processes 1  # Custom process counts
        """
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default="computation_test",
        help='Base path where to save machine info JSON files (default: computation_test)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Duration of computational benchmark in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--cpu-processes',
        type=int,
        default=None,
        help='Number of computation processes (default: number of CPU cores)'
    )
    
    parser.add_argument(
        '--memory-processes',
        type=int,
        default=2,
        help='Number of memory-intensive processes (default: 2)'
    )
    
    parser.add_argument(
        '--info-interval',
        type=int,
        default=10,
        help='Interval in seconds between machine info saves (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("Error: Duration must be positive")
        return 1
    
    if args.memory_processes < 0:
        print("Error: Memory processes must be non-negative")
        return 1
    
    if args.info_interval <= 0:
        print("Error: Info interval must be positive")
        return 1
    
    print("=" * 60)
    print("Computational Heavy Problem Solver with Machine Info Monitoring")
    print("=" * 60)
    print(f"Duration: {args.duration} seconds")
    print(f"Save path: {args.path}")
    print(f"Info interval: {args.info_interval} seconds")
    print(f"Memory processes: {args.memory_processes}")
    if args.cpu_processes:
        print(f"CPU processes: {args.cpu_processes}")
    print("=" * 60)
    
    # Check if machine.py exists
    if not os.path.exists("machine.py"):
        print("Error: machine.py not found in current directory!")
        print("Please run this script from the same directory as machine.py")
        return 1
    
    # Run the computational benchmark
    try:
        run_computational_benchmark(
            duration_seconds=args.duration, 
            save_path=args.path,
            cpu_processes=args.cpu_processes,
            memory_processes=args.memory_processes,
            info_interval=args.info_interval
        )
        return 0
    except Exception as e:
        print(f"Error running computational benchmark: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
