# LLM Benchmark Runner

[![CI Tests](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml)

A command-line tool to benchmark local (Ollama, vLLM) and remote (Google Gemini) Large Language Models (LLMs). It evaluates models against a configurable set of tasks, monitors system resources, generates a detailed HTML report with performance visualizations, and supports exporting results.

This project is designed specifically to help a specific project with testing and benchmarking use cases, scenarios, and prompting styles for use in its own engine. I hope you find it useful for yourself.

## Features

*   **Multi-Provider Support:** Benchmark models served locally via [Ollama](https://ollama.com/), [vLLM](https://github.com/vllm-project/vllm) (OpenAI-compatible API), and remotely via the Google Gemini API.
*   **Flexible Task Definition:** Define benchmark tasks in a simple JSON format (`benchmark_tasks.json`), organized by category. Includes tasks with varied prompts (e.g., different instructions, personas).
*   **Configuration File:** Manage default settings (models, paths, API keys/URLs, weights, features, timeouts) via a `config.yaml` file. CLI arguments and environment variables override file settings.
*   **Diverse Evaluation Methods:**
    *   Keyword matching (strict 'all' or flexible 'any')
    *   Weighted keyword scoring for nuanced evaluation
    *   JSON and YAML structure validation and comparison
    *   Regex-based information extraction with optional validation rules
    *   Python code execution and testing against defined test cases
    *   Classification with confidence score checking
    *   Semantic similarity comparison (optional, requires `sentence-transformers`)
*   **Resource Monitoring (Optional):**
    *   Track CPU RAM usage delta for local models (Ollama, vLLM) (requires `psutil`).
    *   Track NVIDIA GPU memory usage delta (GPU 0) for local models (requires `pynvml`).
*   **Performance Metrics:** Measure API response time (latency) for all providers and tokens/second (Ollama only).
*   **Scoring:** Calculates overall accuracy, average scores for partial credit tasks, an "Ollama Performance Score" (Ollama only), and a category-weighted "Overall Score".
*   **Reporting:**
    *   Generates a comprehensive HTML report with summary tables, performance plots (rankings for scores, accuracy, token/sec (Ollama), resource usage, comparison by prompt stage), and detailed per-task results.
    *   Optional export of summary results to CSV (`--export-summary-csv`).
    *   Optional export of detailed task results to JSON (`--export-details-json`).
*   **Caching:** Caches results to speed up subsequent runs (configurable TTL).
*   **Utilities:** Includes a `--check-dependencies` flag to verify installation and basic functionality of optional libraries.
*   **Configurable:** Control models, tasks, retries, paths, optional features, scoring weights, API endpoints, and more via `config.yaml`, environment variables, and command-line arguments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/colonelpanik/llm-bench.git
    cd llm-bench
    ```

2.  **Recommended: Create and Activate a Virtual Environment:**
    ```bash
    python -m venv .venv
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows:
    # .\.venv\Scripts\activate
    ```

3.  **Install Core Dependencies:**
    The core requirements are `requests`, `PyYAML`, and `openai`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install Optional Dependencies (As Needed):**
    Install libraries for features you intend to use. See `requirements.txt` for details.
    *   **RAM Monitoring (`--ram-monitor enable`):** `pip install psutil`
    *   **GPU Monitoring (`--gpu-monitor enable`):** `pip install pynvml` (Requires NVIDIA drivers/CUDA toolkit correctly installed)
    *   **Report Plots (`--visualizations enable`):** `pip install matplotlib`
    *   **Semantic Evaluation (`--semantic-eval enable`):** `pip install sentence-transformers` (Downloads model files on first use)

    You can check the status of optional dependencies using:
    ```bash
    python -m benchmark_cli --check-dependencies
    ```

## Configuration Precedence

Settings are determined in the following order (later steps override earlier ones):

1.  **Base Defaults:** Hardcoded minimal defaults in `config.py`.
2.  **`config.yaml`:** Settings loaded from the YAML configuration file (default: `config.yaml`, path configurable via `--config-file`). **This is the primary place to set your defaults.** Includes `api.ollama_host_url`, `api.vllm_host_url`, `api.gemini_api_key`, etc.
3.  **Environment Variables:**
    *   `GEMINI_API_KEY` overrides `api.gemini_api_key` from `config.yaml`.
    *   `OLLAMA_HOST` overrides `api.ollama_host_url` from `config.yaml`.
    *   `VLLM_HOST` overrides `api.vllm_host_url` from `config.yaml`.
    *   `VLLM_API_KEY` overrides `api.vllm_api_key` from `config.yaml`.
4.  **Command-Line Arguments:** Any arguments provided on the command line override all previous settings (e.g., `--test-model`, `--tasks-file`, `--gemini-key`, `--vllm-host`, `--ram-monitor disable`).

## Model Identification

*   **Ollama:** Models specified without a prefix (e.g., `llama3:8b`) or with `ollama/` prefix (e.g., `ollama/llama3:8b`).
*   **vLLM:** Models specified with `vllm/` prefix (e.g., `vllm/meta-llama/Llama-2-7b-chat-hf`). The part after the prefix is passed to the vLLM server.
*   **Gemini:** Models specified with `gemini-` prefix (e.g., `gemini-1.5-flash-latest`) or `models/` prefix.

## Usage

The main entry point is `benchmark_cli.py`.

**Show Help:**
```bash
python -m benchmark_cli --help
```

**Basic Run (Uses defaults from `config.yaml` or base):**
```bash
python -m benchmark_cli
```

**Run specific models (Ollama, vLLM, Gemini), overriding config defaults, clear cache:**
```bash
python -m benchmark_cli \
  --test-model ollama/llama3:8b \
  --test-model vllm/meta-llama/Llama-3-8B-Instruct \
  --test-model gemini-1.5-flash-latest \
  --clear-cache -v
```

**Run only 'nlp' tasks category and open report:**
```bash
python -m benchmark_cli --task-set nlp --open-report
```

**Run only specific tasks by name:**
```bash
python -m benchmark_cli --task-name "Sentiment - Complex Complaint" --task-name "Code - Python Factorial"
```

**Set Gemini API Key and vLLM Host via CLI (highest precedence):**
```bash
python -m benchmark_cli \
  --gemini-key "YOUR_GEMINI_KEY" \
  --vllm-host "http://your-vllm-server:8000/v1" \
  --test-model gemini-1.5-pro-latest \
  --test-model vllm/model-on-vllm
```
*(Alternatively, set environment variables or define in `config.yaml`)*

**Use a custom configuration file:**
```bash
python -m benchmark_cli --config-file my_settings.yaml
```

**Run and export summary results to CSV:**
```bash
python -m benchmark_cli --export-summary-csv
```

**Check if optional dependencies are installed and working:**
```bash
python -m benchmark_cli --check-dependencies
```

## Configuration Files

*   **`config.yaml` (Default):** Define default models, API endpoints (`ollama_host_url`, `vllm_host_url`, `gemini_api_key`), paths, weights, timeouts, feature toggles, etc. See the default file for structure and comments.
*   **`benchmark_tasks.json` (Default):** Define your benchmark tasks here. Path configurable in `config.yaml` or via `--tasks-file`.
*   **`report_template.html` (Default):** Customize the HTML report template. Path configurable in `config.yaml` or via `--template-file`.

## Output

*   **HTML Report (`benchmark_report/report.html`):** Detailed report. Path configurable.
*   **Plots (`benchmark_report/images/*.png`):** PNG images embedded in the report. Path configurable.
*   **Cache Files (`benchmark_cache/cache_*.json`):** Stored results. Path configurable.
*   **Export Files (Optional):**
    *   `benchmark_report/summary_*.csv`: Summary CSV file if `--export-summary-csv` is used.
    *   `benchmark_report/details_*.json`: Detailed JSON file if `--export-details-json` is used.
*   **Console Output:** Progress, summaries, warnings, errors. Use `-v` for more detail.

## GitHub Actions / CI

This project uses GitHub Actions for automated testing. Unit tests are run automatically on pushes and pull requests to the `main` branch against multiple Python versions.

[![CI Tests](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml/badge.svg)](https://github.com/colonelpanik/llm-bench/actions/workflows/python-test.yml)

The tests mock external services (Ollama, Gemini, vLLM) and do not require live instances or API keys to run in the CI environment.

## Docker Usage

A `Dockerfile` is provided for running the benchmark tool in a containerized environment with all dependencies included.

**1. Build the Docker Image:**
From the project root directory:
```bash
docker build -t llm-bench .
```

**2. Run the Benchmark:**
Running the benchmark requires connecting the container to your running Ollama and/or vLLM instances. The method depends on your operating system.

*   **On Linux:** Use `--network=host` to share the host's network stack. Services running on `localhost:PORT` on the host will be accessible via the same address inside the container. Mount volumes for configuration, tasks, reports, and cache.
    ```bash
    # Example running Ollama and vLLM models
    docker run --rm -it --network=host \
      -v ./config.yaml:/app/config.yaml \
      -v ./benchmark_tasks.json:/app/benchmark_tasks.json \
      -v ./benchmark_report:/app/benchmark_report \
      -v ./benchmark_cache:/app/benchmark_cache \
      llm-bench \
      --test-model ollama/llama3:8b \
      --test-model vllm/meta-llama/Llama-3-8B-Instruct
    ```
    *(Note: `--open-report` might not work reliably from within Docker unless you have a browser configured.)*

*   **On macOS or Windows (Docker Desktop):** `--network=host` is not typically supported. Instead, Docker provides a special DNS name `host.docker.internal` which resolves to the host machine. You need to tell `llm-bench` to use this address for Ollama/vLLM via environment variables or by modifying `config.yaml`.

    *   **Using Environment Variables:** Set `OLLAMA_HOST` and/or `VLLM_HOST`.
        ```bash
        docker run --rm -it \
          -v ./config.yaml:/app/config.yaml \
          -v ./benchmark_tasks.json:/app/benchmark_tasks.json \
          -v ./benchmark_report:/app/benchmark_report \
          -v ./benchmark_cache:/app/benchmark_cache \
          -e OLLAMA_HOST="http://host.docker.internal:11434" \
          -e VLLM_HOST="http://host.docker.internal:8000/v1" \
          llm-bench \
          --test-model ollama/llama3:8b \
          --test-model vllm/meta-llama/Llama-3-8B-Instruct
        ```

    *   **Modifying `config.yaml`:** Update `api.ollama_host_url` and `api.vllm_host_url` in your mounted `config.yaml` to point to `http://host.docker.internal:PORT`.

**Running Specific Commands:**
You can pass any `llm-bench` command-line arguments after the image name:
```bash
# Check dependencies inside the container
docker run --rm -it llm-bench --check-dependencies

# Run specific models with verbose output (Linux example)
docker run --rm -it --network=host -v $(pwd):/app llm-bench --test-model ollama/mistral:7b -v
```
*(Adjust `--network=host` or add `-e ..._HOST` based on your OS for commands requiring local servers.)*

## License

MIT License.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.
```