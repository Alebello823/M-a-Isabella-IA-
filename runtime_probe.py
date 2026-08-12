import os
import time
import subprocess


def read_meminfo():
    data = {}

    with open("/proc/meminfo", "r") as f:
        for line in f:
            key, value = line.split(":", 1)
            parts = value.strip().split()

            if parts:
                data[key] = int(parts[0])

    return data


def read_cpu_stat():
    with open("/proc/stat", "r") as f:
        line = f.readline()

    values = line.split()[1:]

    return [int(v) for v in values]


def main():
    print("=" * 50)
    print("MÍA ISABELLA - RUNTIME PROBE")
    print("=" * 50)

    mem = read_meminfo()
    cpu = read_cpu_stat()

    print("\n[MEMORY]")
    print(f"MemTotal:     {mem.get('MemTotal', 0) / 1024 / 1024:.2f} GiB")
    print(f"MemAvailable: {mem.get('MemAvailable', 0) / 1024 / 1024:.2f} GiB")
    print(f"SwapTotal:    {mem.get('SwapTotal', 0) / 1024 / 1024:.2f} GiB")
    print(f"SwapFree:     {mem.get('SwapFree', 0) / 1024 / 1024:.2f} GiB")

    print("\n[CPU]")
    print("Raw CPU counters:", cpu[:8])

    print("\n[PROCESS]")
    print("PID:", os.getpid())
    print("Threads:", os.cpu_count())

    print("\n[CONFIG]")
    print("Context: 2048")
    print("Threads: 4")
    print("Max tokens: 256")
    print("Temperature: 0.7")


if __name__ == "__main__":
    main()
