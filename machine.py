import psutil
import platform
import uuid
import os
import time
import json
import subprocess
import getpass
import socket
import shutil
from datetime import datetime
import argparse

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None

# -------- GPU DETECTION --------
def get_nvidia_info():
    if not shutil.which("nvidia-smi"):
        return None
    output = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits"
    ])
    if not output:
        return None
    gpus = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        try:
            name, mem_total, mem_used, mem_free, util_gpu, power_draw, temp = line.split(", ")
            gpus.append({
                "vendor": "NVIDIA",
                "name": name,
                "memory_total_MB": int(mem_total),
                "memory_used_MB": int(mem_used),
                "memory_free_MB": int(mem_free),
                "utilization_percent": int(util_gpu),
                "power_draw_W": float(power_draw),
                "temperature_C": int(temp)
            })
        except Exception:
            continue
    return gpus if gpus else None

def get_amd_info():
    if not shutil.which("rocm-smi"):
        return None
    output = run_cmd(["rocm-smi", "--showid", "--showuse", "--showmeminfo", "vram", "--showpower", "--showtemp"])
    if not output:
        return None
    return [{"vendor": "AMD", "raw_output": output}]

def get_intel_info():
    if shutil.which("intel_gpu_top"):
        output = run_cmd(["intel_gpu_top", "-J", "-s", "1000"])
        if output:
            return [{"vendor": "Intel", "raw_output": output}]
    return None

def get_apple_info():
    if platform.system() != "Darwin":
        return None
    output = run_cmd(["system_profiler", "SPDisplaysDataType", "-json"])
    if not output:
        return None
    try:
        data = json.loads(output)
        gpus = []
        for gpu in data.get("SPDisplaysDataType", []):
            gpus.append({
                "vendor": "Apple",
                "name": gpu.get("sppci_model"),
                "memory": gpu.get("spdisplays_vram") or gpu.get("spdisplays_vram_shared"),
                "metal_support": gpu.get("spdisplays_metal")
            })
        return gpus
    except Exception:
        return None

def get_gpu_info():
    return (
        get_nvidia_info() or
        get_amd_info() or
        get_intel_info() or
        get_apple_info()
    )

# -------- MACHINE STATE --------
def get_machine_state():
    # CPU
    number_of_cpu = psutil.cpu_count(logical=True)
    cpu_utilization = psutil.cpu_percent(interval=1)
    
    # Process count
    total_processes = len(psutil.pids())

    # Memory
    mem = psutil.virtual_memory()
    memory_size = mem.total
    memory_utilization = mem.percent

    # Machine UUID
    try:
        machine_uuid = str(uuid.UUID(int=uuid.getnode()))
    except Exception:
        machine_uuid = "unknown"

    # Power utilization (Linux only, optional)
    power_utilization = None
    try:
        if os.path.exists("/sys/class/power_supply/BAT0/power_now"):
            with open("/sys/class/power_supply/BAT0/power_now") as f:
                power_utilization = int(f.read().strip()) / 1e6  # Watts
    except Exception:
        pass

    # CPU temperature
    cpu_temp = None
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            cpu_temp = {name: [t.current for t in entries] for name, entries in temps.items()}
    except Exception:
        pass

    # Uptime
    uptime_seconds = time.time() - psutil.boot_time()

    # System info
    os_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    # Load average (not available on Windows)
    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        load_avg = None

    # GPU info
    gpu_info = get_gpu_info()

    # Username & Hostname
    username = getpass.getuser()
    hostname = socket.gethostname()

    # Pack results
    machine_state = {
        "username": username,
        "hostname": hostname,
        "number_of_cpu": number_of_cpu,
        "total_processes": total_processes,
        "memory_size": memory_size,
        "cpu_utilization": cpu_utilization,
        "memory_utilization": memory_utilization,
        "machine_uuid": machine_uuid,
        "power_utilization": power_utilization,
        "cpu_temperature": cpu_temp,
        "uptime_seconds": int(uptime_seconds),
        "os_info": os_info,
        "load_avg": load_avg,
        "gpu_info": gpu_info
    }

    return machine_state


def get_machine_info_json():
    """
    Get machine information as a JSON string.
    
    Returns:
        str: JSON string containing machine information
    """
    return json.dumps(get_machine_state(), indent=4)


def get_machine_info_dict():
    """
    Get machine information as a Python dictionary.
    
    Returns:
        dict: Dictionary containing machine information
    """
    return get_machine_state()


def save_machine_info_to_file(base_path=None):
    """
    Save machine information to a JSON file with timestamp in the filename.
    Creates the directory if it doesn't exist.
    
    Args:
        base_path (str, optional): Base directory path to save the file. 
                                 If None, saves to current directory.
    
    Returns:
        str: Full path to the saved file
    """
    # Use current directory if base_path is not provided
    if base_path is None:
        base_path = os.getcwd()
    
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.json"
    
    # Full path to the file
    file_path = os.path.join(base_path, filename)
    
    # Get machine info and save to file
    machine_info_json = get_machine_info_json()
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(machine_info_json)
    
    return file_path


def main():
    """
    Main function to handle CLI arguments and execute machine info collection.
    """
    parser = argparse.ArgumentParser(
        description="Collect and save machine information to JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Print JSON to console
  python main.py --save                    # Save to current directory with timestamp
  python main.py --save --path /tmp/logs   # Save to specific directory
  python main.py --path ./machine_logs     # Save to relative directory
        """
    )
    
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save machine info to a timestamped JSON file'
    )
    
    parser.add_argument(
        '--path', '-p',
        type=str,
        default=None,
        help='Base path where to save the JSON file (implies --save)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output, only save file (useful for scripts)'
    )
    
    args = parser.parse_args()
    
    # If path is provided, automatically enable save mode
    if args.path:
        args.save = True
    
    if args.save:
        # Save to file
        try:
            saved_path = save_machine_info_to_file(args.path)
            if not args.quiet:
                print(f"Machine info saved to: {saved_path}")
                file_size = os.path.getsize(saved_path)
                print(f"File size: {file_size} bytes")
        except Exception as e:
            print(f"Error saving file: {e}")
            return 1
    else:
        # Print to console
        if not args.quiet:
            machine_info_json = get_machine_info_json()
            print(machine_info_json)
    
    return 0


if __name__ == "__main__":
    exit(main())
