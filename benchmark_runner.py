import time
import traceback
import os
from collections import defaultdict
# Import utils functions used directly
from utils import format_na, truncate_text
# Import client and monitor functions
from llm_clients import (
    query_ollama, query_gemini, query_vllm, query_llamacpp, # --- VLLM ADDITION ---
    get_local_ollama_models, get_ollama_running_models, unload_ollama_model,
    get_provider_from_model_name
)
from evaluation import evaluate_response
from system_monitor import get_ollama_pids, get_vllm_pids, get_combined_rss, get_gpu_memory_usage # --- VLLM ADDITION ---

# --- Constants ---
UNLOAD_WAIT_SECONDS = 5 # Time to wait after unload requests before measuring RAM/GPU
INTER_UNLOAD_DELAY = 0.5 # Small delay between unload API calls

def run_benchmark_set(benchmark_name, model_list, tasks, task_categories_map, runtime_config):
    """Runs a set of tasks across specified models and collects results."""
    print(f"\n--- Benchmark Set: {benchmark_name} ---")
    if not model_list: print("[WARN] No models specified."); return {}
    if not tasks: print("[WARN] No tasks selected."); return {}

    results = {}
    task_lookup = {task['name']: task for task in tasks}

    # --- Initial Resource State ---
    # MODIFIED FOR VLLM: Check for any local provider (Ollama or vLLM)
    ollama_models_present = any(get_provider_from_model_name(m) == "ollama" for m in model_list)
    vllm_models_present = any(get_provider_from_model_name(m) == "vllm" for m in model_list)
    llamacpp_models_present = any(get_provider_from_model_name(m) == "llamacpp" for m in model_list)
    local_provider_present = ollama_models_present or vllm_models_present or llamacpp_models_present

    ollama_pids = []
    vllm_pids = [] # --- VLLM ADDITION ---
    combined_pids = []
    initial_ram_bytes = 0
    initial_gpu_mem_bytes = 0

    if local_provider_present and (runtime_config.ram_monitor_enabled or runtime_config.gpu_monitor_enabled):
        # --- Attempt to Unload Existing Ollama Models (Only if Ollama is being tested) ---
        if ollama_models_present:
            print("[INFO] Checking for currently running models in Ollama...")
            running_models_before = get_ollama_running_models(runtime_config)

            if running_models_before is None:
                print("[WARN] Could not determine running Ollama models due to API error. Baseline measurement might be affected.")
            elif not running_models_before:
                print("[INFO] No models appear to be loaded in Ollama memory.")
            else:
                print(f"[INFO] Attempting to unload currently running Ollama models: {running_models_before}")
                unload_success_count = 0
                unload_fail_count = 0
                for model_to_unload in running_models_before:
                    if unload_ollama_model(model_to_unload, runtime_config):
                        unload_success_count += 1
                    else:
                        unload_fail_count += 1
                    time.sleep(INTER_UNLOAD_DELAY) # Small pause between unload calls

                print(f"[INFO] Ollama unload requests sent ({unload_success_count} success, {unload_fail_count} fail). Waiting {UNLOAD_WAIT_SECONDS}s for memory release...")
                time.sleep(UNLOAD_WAIT_SECONDS)

        # --- Measure Baseline ---
        print("[INFO] Measuring initial RAM/GPU state for local providers...")
        if runtime_config.ram_monitor_enabled:
            if runtime_config.psutil_available:
                # Get PIDs for each provider if present
                if ollama_models_present:
                    ollama_pids = get_ollama_pids(runtime_config.psutil, runtime_config)
                if vllm_models_present:
                    vllm_pids = get_vllm_pids(runtime_config.psutil, runtime_config) # --- VLLM ADDITION ---

                combined_pids = list(set(ollama_pids + vllm_pids)) # Unique PIDs

                if combined_pids:
                    initial_ram_bytes = get_combined_rss(combined_pids, runtime_config.psutil)
                    print(f"[INFO] Initial Combined RAM: {format_na(initial_ram_bytes / (1024*1024), ' MB')} (PIDs: {combined_pids})")
                else:
                    print("[WARN] Initial RAM: Could not find PIDs for local providers. RAM Delta metrics will be inaccurate.")
            else:
                 print("[INFO] RAM monitoring enabled but psutil unavailable. Skipping RAM measurement.")

        if runtime_config.gpu_monitor_enabled:
            if runtime_config.pynvml_available and runtime_config.gpu_count > 0:
                initial_gpu_mem_bytes = get_gpu_memory_usage(runtime_config.pynvml, 0)
                print(f"[INFO] Initial GPU 0 Mem: {format_na(initial_gpu_mem_bytes / (1024*1024), ' MB')}")
            elif not runtime_config.pynvml_available:
                print("[INFO] GPU monitoring enabled but PyNVML unavailable/failed init. Skipping GPU measurement.")
            elif runtime_config.gpu_count == 0:
                 print("[INFO] GPU monitoring enabled but no NVIDIA GPUs detected. Skipping GPU measurement.")

    initial_ram_mb = initial_ram_bytes / (1024*1024) if initial_ram_bytes > 0 else 0
    initial_gpu_mem_mb = initial_gpu_mem_bytes / (1024*1024) if initial_gpu_mem_bytes > 0 else 0

    # --- Loop Through Models ---
    local_ollama_models_cache = None # Cache for Ollama model check
    general_models_to_run = set(runtime_config.models_to_benchmark)
    code_models_to_run = set(runtime_config.code_models_to_benchmark)

    for idx, model_name_raw in enumerate(model_list):
        # --- VLLM MODIFICATION: Determine provider and clean model name ---
        provider = get_provider_from_model_name(model_name_raw)
        # Use the raw name for display, cleaned name for API calls if needed
        model_name_for_api = model_name_raw.split(f'{provider}/', 1)[-1] if model_name_raw.startswith(f'{provider}/') else model_name_raw

        print(f"\n>>> Model {idx+1}/{len(model_list)}: {model_name_raw} ({provider}) <<<")

        model_results = {}
        model_summary = {
            "model_name": model_name_raw, # Store the original name with prefix
            "provider": provider, "status": "Pending",
            "accuracy": 0.0, "correct_count": 0, "success_count": 0, "processed_count": 0,
            "api_errors": 0, "error_rate": 0.0, "avg_time_per_task": 0.0, "total_time": 0.0,
            "partial_score_avg": None, "scored_task_count": 0,
            "tokens_per_sec_avg": None, # Applicable mainly to Ollama
            "peak_ram_mb": None, "initial_ram_mb": initial_ram_mb, # Store initial baseline
            "delta_ram_mb": None,
            "peak_gpu_mem_mb": None, "initial_gpu_mem_mb": initial_gpu_mem_mb, # Store initial baseline
            "delta_gpu_mem_mb": None,
            "per_type": defaultdict(lambda: {
                "count": 0, "correct": 0, "api_errors": 0, "score_sum": 0.0, "score_count": 0,
                "time_sum": 0.0, "tokps_sum": 0.0, "tokps_count": 0, "success_api_calls": 0
            })
        }

        # --- Model Availability Check ---
        if provider == "ollama":
            if local_ollama_models_cache is None:
                 local_ollama_models_cache = get_local_ollama_models(runtime_config)
                 if local_ollama_models_cache is None:
                     print(f"  [SKIP MODEL] Cannot verify local Ollama models due to API connection error. Skipping {model_name_raw}.")
                     model_summary["status"] = "Skipped - Ollama API Error"
                     results[model_name_raw] = {"_summary": model_summary}
                     continue
            # Use model_name_for_api for the check
            if model_name_for_api not in local_ollama_models_cache:
                 print(f"  [SKIP MODEL] Ollama model '{model_name_for_api}' not found locally and not pulled. Skipping.")
                 model_summary["status"] = "Skipped - Model Not Found"
                 results[model_name_raw] = {"_summary": model_summary}
                 continue
        elif provider == "gemini" and not runtime_config.gemini_key:
            print(f"  [SKIP MODEL] Gemini model '{model_name_raw}' specified but no API key provided. Skipping.")
            model_summary["status"] = "Skipped - No API Key"
            results[model_name_raw] = {"_summary": model_summary}
            continue
        elif provider == "vllm":
             # Basic check: ensure host URL is configured
             if not runtime_config.vllm_host_url:
                 print(f"  [SKIP MODEL] vLLM model '{model_name_raw}' specified but vLLM Host URL not configured. Skipping.")
                 model_summary["status"] = "Skipped - No vLLM Host"
                 results[model_name_raw] = {"_summary": model_summary}
                 continue
        elif provider == "llamacpp":
            model_path = model_name_for_api
            if not os.path.exists(model_path):
                print(f"  [SKIP MODEL] Llama.cpp model file not found at '{model_path}'. Skipping.")
                model_summary["status"] = "Skipped - Model File Not Found"
                results[model_name_raw] = {"_summary": model_summary}
                continue

        # Per-Model Tracking...
        total_model_time = 0.0
        correct_tasks = 0
        tasks_with_scores = 0
        sum_partial_scores = 0.0
        tasks_processed_count = 0
        model_api_errors = 0
        successful_api_calls = 0
        sum_tokps = 0.0 # Only relevant for Ollama
        count_tokps = 0 # Only relevant for Ollama
        peak_ram_bytes = initial_ram_bytes
        peak_gpu_mem_bytes = initial_gpu_mem_bytes

        # Loop Through Tasks...
        for tidx, task in enumerate(tasks):
            t_name = task["name"]
            t_type = task["type"]
            prompt = task.get("prompt", "")
            is_code_task = t_type in ["code_generation", "code_explanation", "code_debugging"]

            # Task Skipping Logic...
            if not prompt.strip(): continue
            if t_type.startswith("yaml_") and not runtime_config.pyyaml_available: continue
            if task.get("evaluation_method") == "semantic" and not runtime_config.sentence_transformers_available: continue
            # Check against raw model name (with prefix) for filtering
            if is_code_task and model_name_raw not in code_models_to_run: continue
            if not is_code_task and model_name_raw not in general_models_to_run: continue

            # Run Task...
            print(f"  Running Task {tidx+1}/{len(tasks)}: {t_name} ({t_type})")
            if runtime_config.verbose: print(f"    [Verbose] Prompt: {truncate_text(prompt, 200)}...")
            tasks_processed_count += 1
            model_summary["per_type"][t_type]["count"] += 1

            # Resource Monitoring (Before Query) - Use COMBINED PIDs for local providers
            current_ram_before = 0
            current_gpu_before = 0
            if provider in ["ollama", "vllm"]: # Check if it's a local provider
                if runtime_config.ram_monitor_enabled and runtime_config.psutil_available and combined_pids:
                    current_ram_before = get_combined_rss(combined_pids, runtime_config.psutil)
                    peak_ram_bytes = max(peak_ram_bytes, current_ram_before)
                if runtime_config.gpu_monitor_enabled and runtime_config.pynvml_available and runtime_config.gpu_count > 0:
                    current_gpu_before = get_gpu_memory_usage(runtime_config.pynvml, 0)
                    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, current_gpu_before)

            # --- Query Model (Dispatch based on provider) ---
            start_query_time = time.time()
            resp_text, duration_api, tokps_api, error = "", 0.0, None, "Provider not matched"
            if provider == "gemini":
                resp_text, duration_api, tokps_api, error = query_gemini(model_name_for_api, prompt, runtime_config)
            elif provider == "vllm":
                resp_text, duration_api, tokps_api, error = query_vllm(model_name_for_api, prompt, runtime_config) # Pass cleaned name
            elif provider == "llamacpp":
                resp_text, duration_api, tokps_api, error = query_llamacpp(model_name_for_api, prompt, runtime_config)
            else: # Default to Ollama
                resp_text, duration_api, tokps_api, error = query_ollama(model_name_for_api, prompt, runtime_config) # Pass cleaned name
            query_duration = time.time() - start_query_time # Use actual wall-clock time
            total_model_time += query_duration
            model_summary["per_type"][t_type]["time_sum"] += query_duration

            # Resource Monitoring (After Query) - Use COMBINED PIDs
            if provider in ["ollama", "vllm"]:
                if runtime_config.ram_monitor_enabled and runtime_config.psutil_available and combined_pids:
                    current_ram_after = get_combined_rss(combined_pids, runtime_config.psutil)
                    peak_ram_bytes = max(peak_ram_bytes, current_ram_after)
                if runtime_config.gpu_monitor_enabled and runtime_config.pynvml_available and runtime_config.gpu_count > 0:
                    current_gpu_after = get_gpu_memory_usage(runtime_config.pynvml, 0)
                    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, current_gpu_after)

            # Process Result...
            task_result_data = {
                "response": resp_text, "error": error, "duration": query_duration,
                # Only store tok/s if provider is Ollama/Llama.cpp and value is valid
                "tokens_per_sec": tokps_api if provider in ["ollama", "llamacpp"] and tokps_api is not None and tokps_api > 0 else None,
                "metric": None, "details": "N/A", "task_type": t_type
            }

            if error:
                print(f"    [API Error] {error}")
                model_api_errors += 1
                model_summary["per_type"][t_type]["api_errors"] += 1
                task_result_data["details"] = f"API error: {error}"
                task_result_data["metric"] = False
            else:
                successful_api_calls += 1
                model_summary["per_type"][t_type]["success_api_calls"] += 1
                if runtime_config.verbose: print(f"    [Verbose] Response: {truncate_text(resp_text, 300)}...")

                eval_start_time = time.time()
                try:
                    metric, details = evaluate_response(task, resp_text, runtime_config)
                except Exception as eval_e:
                     print(f"    [Internal Eval Error] Task '{t_name}', Model '{model_name_raw}': {eval_e}")
                     print(traceback.format_exc())
                     metric = False
                     details = f"Internal evaluation error: {eval_e}"
                eval_duration = time.time() - eval_start_time

                task_result_data["metric"] = metric
                task_result_data["details"] = details

                is_pass = False
                if isinstance(metric, bool):
                    marker = "PASS" if metric else "FAIL"
                    print(f"    Result: {marker}. {details}")
                    if metric: correct_tasks += 1; model_summary["per_type"][t_type]["correct"] += 1; is_pass = True
                elif isinstance(metric, (float, int)):
                    score = float(metric)
                    is_pass = (score >= runtime_config.passing_score_threshold)
                    marker = "PASS" if is_pass else "FAIL"
                    print(f"    Result: Score={score:.1f}% => {marker} (Threshold: {runtime_config.passing_score_threshold}%). {details}")
                    sum_partial_scores += score
                    tasks_with_scores += 1
                    model_summary["per_type"][t_type]["score_sum"] += score
                    model_summary["per_type"][t_type]["score_count"] += 1
                    if is_pass: correct_tasks += 1; model_summary["per_type"][t_type]["correct"] += 1
                else:
                    print(f"    [WARN] Unexpected metric type: {type(metric).__name__} ({metric}). Treating as FAIL.")
                    task_result_data["metric"] = False
                    task_result_data["details"] += " (Unexpected metric type)"
                    is_pass = False

                # Only accumulate tok/s for Ollama or Llama.cpp
                if provider in ["ollama", "llamacpp"] and task_result_data["tokens_per_sec"]:
                    sum_tokps += task_result_data["tokens_per_sec"]
                    count_tokps += 1
                    model_summary["per_type"][t_type]["tokps_sum"] += task_result_data["tokens_per_sec"]
                    model_summary["per_type"][t_type]["tokps_count"] += 1

            print(f"    Time: {query_duration:.2f}s", end="")
            if task_result_data["tokens_per_sec"]: # Only print if available (Ollama)
                print(f" | Tok/s: {task_result_data['tokens_per_sec']:.1f}")
            else:
                print() # Newline for non-Ollama

            model_results[t_name] = task_result_data
            time.sleep(0.05) # Keep small delay

        # --- Calculate Model Summary Stats ---
        model_summary["status"] = "Completed"
        model_summary["processed_count"] = tasks_processed_count
        model_summary["api_errors"] = model_api_errors
        model_summary["success_count"] = successful_api_calls
        model_summary["correct_count"] = correct_tasks
        if tasks_processed_count > 0:
             model_summary["avg_time_per_task"] = total_model_time / tasks_processed_count
             model_summary["error_rate"] = (model_api_errors / tasks_processed_count) * 100 if tasks_processed_count > 0 else 0.0
        if successful_api_calls > 0:
            model_summary["accuracy"] = (correct_tasks / successful_api_calls) * 100
        else:
             model_summary["accuracy"] = 0.0
        if tasks_with_scores > 0:
            model_summary["partial_score_avg"] = sum_partial_scores / tasks_with_scores
        model_summary["scored_task_count"] = tasks_with_scores
        model_summary["total_time"] = total_model_time

        # Provider-specific metrics
        if provider in ["ollama", "llamacpp"]:
            if count_tokps > 0: model_summary["tokens_per_sec_avg"] = sum_tokps / count_tokps
            # RAM/GPU handled below for all local providers

        # RAM/GPU metrics for local providers
        if provider in ["ollama", "vllm"]:
            if runtime_config.ram_monitor_enabled and runtime_config.psutil_available:
                 model_summary["peak_ram_mb"] = max(initial_ram_mb, peak_ram_bytes / (1024*1024))
                 model_summary["delta_ram_mb"] = model_summary["peak_ram_mb"] - initial_ram_mb if model_summary["peak_ram_mb"] is not None else None
            if runtime_config.gpu_monitor_enabled and runtime_config.pynvml_available:
                 model_summary["peak_gpu_mem_mb"] = max(initial_gpu_mem_mb, peak_gpu_mem_bytes / (1024*1024))
                 model_summary["delta_gpu_mem_mb"] = model_summary["peak_gpu_mem_mb"] - initial_gpu_mem_mb if model_summary["peak_gpu_mem_mb"] is not None else None

        # Finalize per-type averages...
        for t_type, stats in model_summary["per_type"].items():
             count = stats["count"]
             success_calls = stats["success_api_calls"]
             if count > 0:
                 stats["avg_time"] = stats["time_sum"] / count if count > 0 else 0.0
                 stats["accuracy"] = (stats["correct"] / success_calls) * 100 if success_calls > 0 else 0.0
                 stats["avg_score"] = stats["score_sum"] / stats["score_count"] if stats["score_count"] > 0 else None
                 # Only calculate avg_tokps if it's Ollama or Llama.cpp
                 if provider in ["ollama", "llamacpp"] and stats["tokps_count"] > 0:
                     stats["avg_tokps"] = stats["tokps_sum"] / stats["tokps_count"]
                 else:
                     stats["avg_tokps"] = None

        model_results["_summary"] = model_summary
        results[model_name_raw] = model_results # Use raw name as key

        # Print Model Summary...
        print(f"\n  --- Model Summary: {model_name_raw} ({provider}) ---")
        print(f"    Status: {model_summary['status']}")
        print(f"    Tasks Processed: {tasks_processed_count}, API Errors: {model_api_errors} ({model_summary['error_rate']:.1f}%)")
        print(f"    Accuracy (on {successful_api_calls} success calls): {model_summary['accuracy']:.1f}% ({correct_tasks} correct)")
        if model_summary['partial_score_avg'] is not None:
            print(f"    Avg Score (on {tasks_with_scores} scored tasks): {model_summary['partial_score_avg']:.1f}%")
        print(f"    Avg Time/Task: {model_summary['avg_time_per_task']:.2f}s (Total: {total_model_time:.1f}s)")
        # Provider-specific print
        if provider in ["ollama", "llamacpp"]:
            print(f"    Avg Tok/s (on {count_tokps} tasks): {format_na(model_summary['tokens_per_sec_avg'], precision=1)}")
        # RAM/GPU print for local providers
        if provider in ["ollama", "vllm"]:
            if runtime_config.ram_monitor_enabled: print(f"    Peak RAM: {format_na(model_summary['peak_ram_mb'], ' MB')} (Delta vs initial: {format_na(model_summary['delta_ram_mb'], ' MB')})")
            if runtime_config.gpu_monitor_enabled: print(f"    Peak GPU Mem: {format_na(model_summary['peak_gpu_mem_mb'], ' MB')} (Delta vs initial: {format_na(model_summary['delta_gpu_mem_mb'], ' MB')})")

        # Delay before next model...
        model_transition_delay = getattr(runtime_config, 'model_transition_delay', 1)
        if idx < len(model_list) - 1:
            print(f"\n... Delaying {model_transition_delay}s before next model ...")
            time.sleep(model_transition_delay)

    print(f"\n--- Finished Benchmark Set: {benchmark_name} ---")

    return results
