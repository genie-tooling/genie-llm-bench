# NAME

llmbench - LLM Benchmark Runner

# SYNOPSIS

**llmbench** \[OPTIONS]...

# DESCRIPTION

**llmbench** is a command-line tool designed to benchmark Large Language Models (LLMs). It supports testing locally hosted models via Ollama and vLLM (OpenAI-compatible API), and remote models via the Google Gemini API.

The tool runs a defined set of tasks against specified models, evaluating responses based on various criteria. It monitors system resources (CPU RAM, optionally NVIDIA GPU memory) during local runs (Ollama, vLLM) and generates a comprehensive HTML report summarizing accuracy, performance metrics (latency, tokens/second for Ollama), resource usage, and detailed task results.

Configuration is managed through a combination of a YAML configuration file (`config.yaml` by default), environment variables, and command-line arguments, allowing for flexible setup and execution.

# CONFIGURATION PRECEDENCE

Settings are determined in the following order, with later steps overriding earlier ones:

1.  **Base Defaults:** Minimal hardcoded defaults in the application.
2.  **YAML Configuration File:** Settings loaded from the file specified by **--config-file** (defaults to `config.yaml`). This is the primary way to set defaults for models, paths, weights, API keys/URLs, etc. (See `config.yaml` for keys like `api.ollama_host_url`, `api.vllm_host_url`, `api.gemini_api_key`).
3.  **Environment Variables:**
    *   `GEMINI_API_KEY` overrides `api.gemini_api_key`.
    *   `OLLAMA_HOST` overrides `api.ollama_host_url`.
    *   `VLLM_HOST` overrides `api.vllm_host_url`.
    *   `VLLM_API_KEY` overrides `api.vllm_api_key`.
4.  **Command-Line Arguments:** Arguments provided directly on the command line take the highest precedence.

# MODEL IDENTIFICATION

Models are identified by provider based on their name:
*   **Ollama:** No prefix (e.g., `llama3:8b`) or `ollama/` prefix (e.g., `ollama/llama3:8b`).
*   **vLLM:** `vllm/` prefix (e.g., `vllm/model-id`). The part after the prefix is sent to the vLLM server.
*   **Gemini:** `gemini-` prefix (e.g., `gemini-1.5-flash-latest`) or `models/` prefix.

# OPTIONS

**--config-file** *PATH*
:   Path to the YAML configuration file to load. (Default: `config.yaml`)

**--task-set** *{all,nlp,code,other}*
:   Select which category of tasks to run. (Default: `all`)

**--test-model** *MODEL_NAME*
:   Specify a model name (e.g., `ollama/llama3:8b`, `vllm/model-id`, `gemini-1.5-flash`) to include. Use prefixes as described above. Overrides `default_models` list in config. Can be used multiple times.

**--code-model** *MODEL_NAME*
:   Specify model name ONLY for code tasks. Overrides `code_models` in config. Can be used multiple times.

**--gemini-key** *API_KEY*
:   API key for Google Gemini API calls. Overrides ENV and config.

**--vllm-host** *URL*
:   URL for the vLLM OpenAI-compatible API endpoint (e.g., `http://localhost:8000/v1`). Overrides ENV and config.

**--pull-ollama-models**
:   Attempt to pull required Ollama models if not found locally.

**--no-cache**
:   Force a fresh benchmark run, ignoring cached results.

**--clear-cache**
:   Delete the cache file before running.

**--benchmark-name** *NAME*
:   Descriptive name for the run (report title, cache filename). (Default: "LLM Benchmark Run")

**--open-report**
:   Automatically open the generated HTML report in the browser.

**--tasks-file** *PATH*
:   Path to the JSON task definitions file. Overrides config.

**--report-dir** *PATH*
:   Directory for HTML report and images. Overrides config.

**--cache-dir** *PATH*
:   Directory for cache files. Overrides config.

**--template-file** *PATH*
:   Path to the HTML report template file. Overrides config.

**--retries** *N*
:   Number of API call retries on transient errors. Overrides config.

**--retry-delay** *SECONDS*
:   Delay between retries. Overrides config.

**--category-weights** *JSON_STRING*
:   JSON mapping task categories to weights for scoring. Overrides config. Example: `'{"General NLP": 1.0, "Code Generation": 0.8, "default": 0.5}'`.

**--ram-monitor** *{enable,disable}*
:   Enable/disable CPU RAM usage monitoring (Ollama/vLLM). Requires `psutil`. Overrides config.

**--gpu-monitor** *{enable,disable}*
:   Enable/disable NVIDIA GPU memory usage monitoring (Ollama/vLLM). Requires `pynvml`. Overrides config.

**--visualizations** *{enable,disable}*
:   Enable/disable plot generation in the report. Requires `matplotlib`. Overrides config.

**--semantic-eval** *{enable,disable}*
:   Enable/disable semantic similarity evaluation. Requires `sentence-transformers`. Overrides config.

**--export-summary-csv**
:   Export summary results to a CSV file in the report directory.

**--export-details-json**
:   Export detailed task results (no summaries) to a JSON file.

**--check-dependencies**
:   Check optional dependencies (psutil, pynvml, matplotlib, sentence-transformers, openai) and exit.

**-v**, **--verbose**
:   Enable verbose logging.

**-h**, **--help**
:   Show the help message and exit.

# EXAMPLES

**Run benchmark with defaults from config.yaml:**
```bash
python -m benchmark_cli
```

**Run specific Ollama and vLLM models, overriding config, clearing cache:**
```bash
python -m benchmark_cli \
  --test-model ollama/llama3:8b \
  --test-model vllm/meta-llama/Llama-3-8B-Instruct \
  --vllm-host http://192.168.1.100:8000/v1 \
  --clear-cache -v
```

**Run only code tasks against a specific vLLM code model:**
```bash
python -m benchmark_cli \
  --task-set code \
  --test-model vllm/codellama/CodeLlama-13b-Instruct-hf \
  --open-report
```

# FILES

**config.yaml (Default)**
:   Primary YAML configuration file.

**benchmark_tasks.json (Default, configurable)**
:   JSON file defining benchmark tasks.

**report_template.html (Default, configurable)**
:   HTML template for report generation.

**benchmark_report/report.html (Default, configurable)**
:   Generated HTML report file.

**benchmark_report/images/*.png (Default, configurable)**
:   Image files for plots.

**benchmark_cache/cache_*.json (Default, configurable)**
:   Cache files storing results.

# ENVIRONMENT VARIABLES

**GEMINI_API_KEY**
:   Overrides Gemini API key from config.

**OLLAMA_HOST**
:   Overrides Ollama host URL from config.

**VLLM_HOST**
:   Overrides vLLM host URL from config.

**VLLM_API_KEY**
:   Overrides vLLM API key from config (if needed).

# BUGS

Report bugs via GitHub issues for the project.

# AUTHOR

[Your Name or Organization]
