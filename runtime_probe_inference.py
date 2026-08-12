import os
import time
import subprocess
import threading


MODEL = os.path.expanduser(
    "~/MiaIsabella/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

N_CTX = 2048
N_THREADS = 4
MAX_TOKENS = 64
TEMPERATURE = 0.7


def read_meminfo():
    data = {}

    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                parts = value.strip().split()

                if parts:
                    data[key] = int(parts[0])

    except PermissionError:
        return {}

    return data


def read_process(pid):
    path = f"/proc/{pid}/status"

    result = {}

    try:
        with open(path, "r") as f:
            for line in f:
                if ":" not in line:
                    continue

                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()

    except (FileNotFoundError, PermissionError):
        return None

    return result


def monitor(pid, interval=0.5):
    samples = []

    while True:
        time.sleep(interval)

        process = read_process(pid)

        if process is None:
            break

        rss = process.get("VmRSS", "unknown")
        vm_size = process.get("VmSize", "unknown")
        threads = process.get("Threads", "unknown")

        sample = {
            "rss": rss,
            "vm_size": vm_size,
            "threads": threads,
        }

        samples.append(sample)

        print(
            "[SAMPLE] "
            f"RSS={rss} "
            f"VmSize={vm_size} "
            f"Threads={threads}"
        )

    return samples


def main():

    print("=" * 60)
    print("MÍA ISABELLA - REAL INFERENCE RUNTIME PROBE")
    print("=" * 60)

    print("\n[CONFIG]")
    print("Model:", MODEL)
    print("Context:", N_CTX)
    print("Threads:", N_THREADS)
    print("Max tokens:", MAX_TOKENS)
    print("Temperature:", TEMPERATURE)

    print("\n[BASELINE MEMORY]")

    mem = read_meminfo()

    if mem:
        print(
            f"MemAvailable: "
            f"{mem.get('MemAvailable', 0) / 1024 / 1024:.2f} GiB"
        )

        print(
            f"SwapFree: "
            f"{mem.get('SwapFree', 0) / 1024 / 1024:.2f} GiB"
        )
    else:
        print("Meminfo no disponible.")

    command = [
        "llama-cli",
        "-m",
        MODEL,
        "-c",
        str(N_CTX),
        "-t",
        str(N_THREADS),
        "-n",
        str(MAX_TOKENS),
        "--temp",
        str(TEMPERATURE),
        "-p",
        "Explica brevemente por qué un sistema puede generar texto lentamente.",
    ]

    print("\n[COMMAND]")
    print(" ".join(command))

    print("\n[START INFERENCE]")

    start = time.time()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    monitor_thread = threading.Thread(
        target=monitor,
        args=(process.pid,),
        daemon=True,
    )

    monitor_thread.start()

    stdout, stderr = process.communicate()

    elapsed = time.time() - start

    monitor_thread.join(timeout=2)

    print("\n[RESULT]")

    print("Return code:", process.returncode)
    print(f"Elapsed: {elapsed:.2f} seconds")

    if stdout:
        print("\n[OUTPUT]")
        print(stdout[-1500:])

    if stderr:
        print("\n[STDERR]")
        print(stderr[-3000:])

    print("\n[DONE]")


if __name__ == "__main__":
    main()
