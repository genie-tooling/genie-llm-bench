# --- System Resource Monitoring (RAM, GPU) ---

import re
# Removed direct requests/json/config imports, use fallback from llm_clients
from llm_clients import _get_ollama_pid_via_api

# --- RAM Monitoring ---

def get_ollama_pids(psutil_module, runtime_config):
    """Identifies PIDs for Ollama processes using psutil, with API fallback."""
    pids = []
    if not psutil_module:
        print("[WARN] psutil not available, cannot find Ollama PIDs via process iteration.")
        # Try API fallback even if psutil isn't loaded
        api_pid = _get_ollama_pid_via_api(runtime_config)
        if api_pid:
            print(f"[INFO] Found potential Ollama PID {api_pid} via API fallback (cannot verify without psutil).")
            pids.append(api_pid)
        return pids

    # psutil is available, try process iteration first
    try:
        NoSuchProcess = getattr(psutil_module, 'NoSuchProcess', Exception)
        AccessDenied = getattr(psutil_module, 'AccessDenied', Exception)

        for proc in psutil_module.process_iter(['pid', 'name', 'cmdline']):
            try:
                if not proc.is_running(): continue
                info = proc.info
                pname = info.get('name', '').lower() if info else ''
                cmd = ' '.join(info.get('cmdline', [])).lower() if info and info.get('cmdline') else ''

                # Match common Ollama process names/command lines
                is_ollama = False
                if 'ollama' in pname and 'serve' in cmd: # Be more specific
                    is_ollama = True
                elif 'ollama' in cmd and ('serve' in cmd or '/ollama' in cmd): # Match executable path too
                    is_ollama = True

                if is_ollama:
                    pid = info.get('pid') if info else None
                    if pid and pid not in pids:
                        pids.append(pid)
            except (NoSuchProcess, AccessDenied):
                continue
            except Exception as inner_e:
                 print(f"[WARN] Error inspecting process {getattr(proc, 'pid', 'N/A')} for Ollama: {inner_e}")
    except Exception as e:
        print(f"[WARN] Error iterating processes with psutil for Ollama: {e}")

    # Fallback using ollama show API if psutil fails or finds nothing
    if not pids:
        print("[INFO] No Ollama process found via psutil iteration, trying API fallback...")
        api_pid = _get_ollama_pid_via_api(runtime_config)
        if api_pid:
            if psutil_module.pid_exists(api_pid):
                print(f"[INFO] Found potential Ollama server PID {api_pid} via API and verified with psutil.")
                pids.append(api_pid)
            else:
                 print(f"[WARN] PID {api_pid} from Ollama API error does not seem to exist according to psutil.")
        else:
            print("[INFO] Ollama API fallback did not yield a PID.")

    if not pids:
         print("[WARN] Could not determine Ollama process PIDs for RAM monitoring.")
    return pids

# --- VLLM ADDITION START ---
def get_vllm_pids(psutil_module, runtime_config):
    """Identifies PIDs for vLLM server processes using psutil."""
    pids = []
    if not psutil_module:
        print("[WARN] psutil not available, cannot find vLLM PIDs.")
        return pids

    # Patterns to match vLLM server launch commands
    # These might need adjustment based on how the server is started
    vllm_patterns = [
        r'python.*vllm\.entrypoints\.openai\.api_server', # Common launch method
        r'python.*vllm\.entrypoints\.api_server',         # Alternative entrypoint
        # Add more patterns if needed (e.g., direct script execution)
    ]

    try:
        NoSuchProcess = getattr(psutil_module, 'NoSuchProcess', Exception)
        AccessDenied = getattr(psutil_module, 'AccessDenied', Exception)

        for proc in psutil_module.process_iter(['pid', 'name', 'cmdline']):
            try:
                if not proc.is_running(): continue
                info = proc.info
                cmdline = ' '.join(info.get('cmdline', [])).lower() if info and info.get('cmdline') else ''

                if not cmdline: continue # Skip processes without command line info

                for pattern in vllm_patterns:
                    if re.search(pattern, cmdline):
                        pid = info.get('pid')
                        if pid and pid not in pids:
                            pids.append(pid)
                            # Found a match for this process, no need to check other patterns for it
                            break
            except (NoSuchProcess, AccessDenied):
                continue
            except Exception as inner_e:
                 print(f"[WARN] Error inspecting process {getattr(proc, 'pid', 'N/A')} for vLLM: {inner_e}")
    except Exception as e:
        print(f"[WARN] Error iterating processes with psutil for vLLM: {e}")

    if not pids:
         print("[WARN] Could not determine vLLM process PIDs for RAM monitoring. Check launch command and patterns.")
    return pids
# --- VLLM ADDITION END ---


def get_combined_rss(pids, psutil_module):
    """Calculates the total RSS memory usage for a list of PIDs."""
    if not psutil_module or not pids:
        return 0
    total_rss = 0
    active_pids_found = [] # Track PIDs that were successfully measured
    NoSuchProcess = getattr(psutil_module, 'NoSuchProcess', Exception)
    AccessDenied = getattr(psutil_module, 'AccessDenied', Exception)

    for pid in pids:
        try:
            p = psutil_module.Process(pid)
            if p.is_running(): # Check if running before accessing memory_info
                mem_info = p.memory_info()
                total_rss += mem_info.rss
                active_pids_found.append(pid)
        except NoSuchProcess:
            if pid in active_pids_found: # If it was active before, maybe log disappearance?
                 print(f"[INFO] Process PID {pid} disappeared during memory measurement.")
                 active_pids_found.remove(pid)
            # else: Process likely ended before measurement started, ignore silently
        except AccessDenied:
            print(f"[WARN] Access denied when getting memory for PID {pid}. RAM delta may be inaccurate.")
            # Keep PID in case permissions change, but don't add to active_pids_found for *this* measurement
        except Exception as e:
            print(f"[WARN] Error getting memory for PID {pid}: {e}")
            # Keep PID, maybe temporary issue

    # Report if some PIDs couldn't be measured
    if len(pids) > len(active_pids_found):
         missed_pids = [p for p in pids if p not in active_pids_found]
         # print(f"[INFO] Could not get memory info for PIDs: {missed_pids} (ended or access denied).")

    return total_rss


# --- GPU Monitoring (NVIDIA Only) ---

def get_gpu_memory_usage(pynvml_module, device_index=0):
    """Gets the used memory for a specific NVIDIA GPU."""
    if not pynvml_module:
        return 0
    try:
        NVMLError = getattr(pynvml_module, 'NVMLError', Exception) # Get NVMLError safely
        handle = pynvml_module.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml_module.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used
    except NVMLError as e:
        # Reduce noise: only print specific errors once? Maybe later.
        # print(f"[WARN] Failed to get GPU memory info for device {device_index}: {e}")
        return 0
    except IndexError:
        # print(f"[WARN] GPU device index {device_index} out of range.")
        return 0 # device_index is invalid
    except Exception as e:
        print(f"[WARN] Unexpected error getting GPU memory for device {device_index}: {e}")
        return 0
