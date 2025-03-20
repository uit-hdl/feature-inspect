from fi_misc.global_util import logger
import os
import threading
import time
from datetime import timedelta
from functools import wraps

import psutil


def monitor_gpu(stop_event, interval, action, writer, gpu_index=0):
    import importlib.util

    pynvml_module = importlib.util.find_spec("pynvml")
    if pynvml_module is not None:
        from pynvml import (
            nvmlInit,
            nvmlShutdown,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetUtilizationRates,
        )
    else:
        return
    """
    Periodically logs GPU memory and processor usage.

    Args:
        stop_event: A threading.Event that signals the monitor to stop.
        interval: Time in seconds between each log.
        action: A string identifier for the action being monitored.
        writer: A logging writer (e.g., TensorBoard writer).
        gpu_index: Index of the GPU to monitor (default: 0).
    """
    try:
        nvmlInit()
    except Exception as e:
        logger.warning(
            "Failed to initialize NVML: '%s'. This disables GPU monitoring. This message is safe to ignore",
            e,
        )
        return
    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        i = 0
        while not stop_event.is_set():
            # Memory usage
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_used = mem_info.used / (1024 * 1024)  # Convert to MB
            gpu_mem_total = mem_info.total / (1024 * 1024)  # Convert to MB

            # GPU utilization
            utilization = nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu  # GPU utilization percentage

            # Log metrics
            writer.add_scalar(f"{action}_gpu_mem_used_mb", gpu_mem_used, global_step=i)
            writer.add_scalar(
                f"{action}_gpu_mem_total_mb", gpu_mem_total, global_step=i
            )
            writer.add_scalar(f"{action}_gpu_util_percent", gpu_util, global_step=i)
            i += 1

            time.sleep(interval)
    finally:
        nvmlShutdown()


def monitor_memory(stop_event, interval, action, writer):
    """Periodically logs memory usage of the current process."""
    process = psutil.Process(os.getpid())
    i = 0
    while not stop_event.is_set():
        mem_info = process.memory_info()
        mem_used = mem_info.rss / (1024 * 1024)
        writer.add_scalar(f"{action}_mem_mb", mem_used, global_step=i)
        i += 1
        time.sleep(interval)


def track_method(func, action_name, writer, track_gpu=False, i=0):
    if i == -1:
        i = 0
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not writer:
            return func(*args, **kwargs)
        start = time.time()
        logger.debug("Performing action: %s", action_name)
        stop_event = threading.Event()
        memory_tracker = threading.Thread(
            target=monitor_memory,
            args=(stop_event, 0.5, action_name, writer),
            daemon=True,
        )
        gpu_monitor = None
        if track_gpu:
            gpu_monitor = threading.Thread(
                target=monitor_gpu,
                args=(stop_event, 0.5, action_name, writer),
                daemon=True,
            )

        memory_tracker.start()
        if track_gpu:
            gpu_monitor.start()

        result = func(*args, **kwargs)

        end = time.time() - start
        writer.add_scalar(f"{action_name}_sec", end, i)

        elapsed_time = timedelta(seconds=end)
        days = elapsed_time.days
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = f"{days}d {hours:02}h {minutes:02}m {seconds:02}s"
        writer.add_text(f"{action_name}_sec_str", formatted_time, i)
        stop_event.set()
        memory_tracker.join()
        if track_gpu:
            gpu_monitor.join()
        logger.debug("Action done: %s", action_name)

        return result

    return wrapper
