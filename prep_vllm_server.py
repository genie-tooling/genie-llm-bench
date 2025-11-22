#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
import sys
import yaml
import shlex

# Default config path relative to this script or project root
DEFAULT_CONFIG_FILE = pathlib.Path('config.yaml')
DEFAULT_VLLM_PORT = 8000 # Common default, but can be overridden

def generate_vllm_commands(config_path, gguf_model_path_arg):
    """
    Reads the llm-bench config file, identifies vLLM models,
    and generates example vLLM server launch commands.
    If gguf_model_path_arg is provided, it uses that path for the --model parameter.
    """
    config_path = pathlib.Path(config_path)
    if not config_path.is_file():
        print(f"Error: Config file not found at '{config_path}'", file=sys.stderr)
        sys.exit(1)

    gguf_model_path = None
    if gguf_model_path_arg:
        gguf_model_path = pathlib.Path(gguf_model_path_arg).resolve() # Get absolute path
        if not gguf_model_path.is_file():
            print(f"Error: Provided GGUF model path does not exist or is not a file: '{gguf_model_path}'", file=sys.stderr)
            sys.exit(1)
        print(f"--- Using provided GGUF model path: {gguf_model_path} ---")
    else:
        print("--- No specific GGUF path provided via CLI. Using model IDs from config. ---")


    print(f"--- Reading llm-bench config: {config_path} ---")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        if not isinstance(config_data, dict):
            print(f"Error: Config file '{config_path}' is not a valid dictionary.", file=sys.stderr)
            sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config file '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config file '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Extract relevant info
    default_models = config_data.get('default_models', [])
    code_models = config_data.get('code_models', [])
    api_config = config_data.get('api', {})
    # Extract host and port from vllm_host_url for the command generation
    vllm_host_url = api_config.get('vllm_host_url', f'http://localhost:{DEFAULT_VLLM_PORT}/v1')

    try:
        # Attempt to parse host and port (basic parsing)
        url_parts = vllm_host_url.split(':')
        vllm_host = url_parts[1].strip('/') if len(url_parts) > 1 else '0.0.0.0' # Default to listen on all interfaces
        vllm_port_str = url_parts[2].split('/')[0] if len(url_parts) > 2 else str(DEFAULT_VLLM_PORT)
        vllm_port = int(vllm_port_str)
    except Exception:
        print(f"Warning: Could not accurately parse host/port from '{vllm_host_url}'. Using defaults for command generation (Host: 0.0.0.0, Port: {DEFAULT_VLLM_PORT}).")
        vllm_host = '0.0.0.0'
        vllm_port = DEFAULT_VLLM_PORT


    all_models = default_models + code_models
    vllm_models_in_config = set() # Store the original vllm/ names from config

    print("\n--- Identifying vLLM models specified in config ---")
    for model_name in all_models:
        if isinstance(model_name, str) and model_name.startswith('vllm/'):
            model_id_from_config = model_name.split('vllm/', 1)[-1]
            if model_id_from_config:
                vllm_models_in_config.add(model_name) # Store the full vllm/ name
                print(f"  Found: {model_name} (Config ID: {model_id_from_config})")
            else:
                print(f"  Warning: Skipping invalid vLLM model name format: {model_name}")

    if not vllm_models_in_config and not gguf_model_path:
        print("\nNo models with the 'vllm/' prefix found in the configuration, and no --gguf-model-path provided.")
        print("Add models like 'vllm/your-model-repo/your-model-name' or provide a GGUF path.")
        sys.exit(0)
    elif not vllm_models_in_config and gguf_model_path:
         print("\nWarning: No 'vllm/' models found in config, but a GGUF path was provided.")
         print("Generating a command based *only* on the provided GGUF path.")
         # Add a dummy entry to generate at least one command
         vllm_models_in_config.add(f"vllm/{gguf_model_path.name}")


    print(f"\n--- Example vLLM Server Launch Commands ---")
    print("# NOTE: These are examples! You MUST customize them based on your hardware and needs.")
    print("# Common parameters to consider adding/adjusting:")
    print("#   --tensor-parallel-size N   (Number of GPUs for tensor parallelism)")
    print("#   --gpu-memory-utilization 0.XX (Fraction of GPU memory to use, e.g., 0.90)")
    print("#   --quantization [awq|gptq|squeezellm|gguf|...] (Specify if needed, often inferred for GGUF)")
    print("#   --dtype [auto|half|bfloat16|float] (Data type)")
    print("#   --max-model-len N (Max sequence length)")
    print("#   --enforce-eager (May be needed for some models/hardware)")
    print("#   --swap-space N (Host memory swap in GiB)")
    print("# See vLLM documentation for all options: https://docs.vllm.ai/en/latest/getting_started/quickstart.html")
    print("-" * 60)

    # Generate a command for each unique vLLM model name found in the config
    for model_config_name in sorted(list(vllm_models_in_config)):
        model_id_from_config = model_config_name.split('vllm/', 1)[-1]

        # Determine the value for the --model argument
        if gguf_model_path:
            model_arg_value = str(gguf_model_path) # Use the absolute path from CLI arg
            print(f"# Command generated for config entry '{model_config_name}', using GGUF path: {gguf_model_path}")
        else:
            model_arg_value = model_id_from_config # Use the ID from the config file
            print(f"# Command generated for config entry '{model_config_name}', using config ID: {model_id_from_config}")


        # Basic command using OpenAI entrypoint
        command_parts = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_arg_value, # Use the determined path or ID
            "--host", vllm_host,
            "--port", str(vllm_port),
            # Add common desirable defaults you might want
            "--trust-remote-code", # Often needed for custom architectures/tokenizers from HF
            # "--gpu-memory-utilization", "0.90", # Example: Use 90% GPU memory
            # "--tensor-parallel-size", "1", # Example: Use 1 GPU
        ]
        # If using GGUF, vLLM might need the quantization type explicitly sometimes,
        # but often infers it. Add a note.
        if gguf_model_path:
            print("# (Note: For GGUF, vLLM often infers quantization, but you might need --quantization gguf if issues arise)")


        # Use shlex.join for proper quoting if needed, though basic examples are fine
        command_str = " ".join(shlex.quote(part) for part in command_parts)

        print(command_str)
        print("-" * 60)

    print("\nRemember to run the appropriate command(s) in a separate terminal")
    print(f"before starting llm-bench targeting vLLM at {vllm_host_url}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate example vLLM server launch commands based on llm-bench config."
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_FILE),
        help=f"Path to the llm-bench YAML configuration file (default: {DEFAULT_CONFIG_FILE})"
    )
    # --- GGUF ADDITION ---
    parser.add_argument(
        "--gguf-model-path",
        default=None,
        help="Optional: Absolute or relative path to a specific local .gguf model file. "
             "If provided, this path will be used for the '--model' argument in the generated command(s), "
             "overriding the model ID derived from the config file's 'vllm/' entries."
    )
    # --- GGUF ADDITION END ---

    args = parser.parse_args()
    generate_vllm_commands(args.config_file, args.gguf_model_path)