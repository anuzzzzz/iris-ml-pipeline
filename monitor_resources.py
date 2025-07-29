import psutil
import time
import json

def monitor_system():
    """Monitor system resources during load test"""
    print("🖥️ SYSTEM MONITORING")
    print("=" * 50)
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU Usage: {cpu_percent}%")
    
    # Memory usage
    memory = psutil.virtual_memory()
    print(f"Memory Usage: {memory.percent}%")
    print(f"Available Memory: {memory.available / (1024**3):.1f} GB")
    
    # API process
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        if 'python' in proc.info['name'] and 'api' in ' '.join(proc.cmdline() if proc.cmdline() else []):
            print(f"API Process CPU: {proc.info['cpu_percent']}%")
            print(f"API Process Memory: {proc.info['memory_percent']:.1f}%")
            break

if __name__ == "__main__":
    monitor_system()
