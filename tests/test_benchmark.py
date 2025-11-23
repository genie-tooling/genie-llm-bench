# -*- coding: utf-8 -*-
# --- Unit Tests for LLM Benchmark Runner ---

import unittest
import json
import time
import sys
import io
import pathlib
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock, mock_open, ANY
import requests
import yaml
import re
import openai

# Add project root to path to allow importing modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Modules to test (import after adding to path)
import config
from config import RuntimeConfig, DEFAULT_OLLAMA_BASE_URL, DEFAULT_GEMINI_API_URL_BASE, DEFAULT_VLLM_BASE_URL # Import class and defaults
import utils
from utils import TimeoutException, HAS_SIGNAL_ALARM, truncate_text
import llm_clients # Import module itself
import system_monitor
import evaluation # Imports submodules
from evaluation import standard_evaluators, structured_evaluators, code_evaluator, semantic_evaluator
import scoring
import cache_manager
import reporting
import benchmark_cli # Import the module itself for testing main entry point
from llm_clients import get_provider_from_model_name

# --- Constants for Tests ---
TEST_MODEL_OLLAMA = "ollama/test-ollama-model:latest" # Use prefix for clarity
TEST_MODEL_GEMINI = "gemini-test-model"
TEST_MODEL_VLLM = "vllm/test-vllm-model"
TEST_BENCHMARK_NAME = "Unit Test Benchmark"
DEFAULT_TEST_CONFIG_PATH = pathlib.Path("./config.yaml") # Used in CLI tests

# --- Mock Responses ---
MOCK_OLLAMA_RESPONSE_SUCCESS = {
    "model": TEST_MODEL_OLLAMA.split('/')[-1], "created_at": "2023-10-27T14:00:00.000Z",
    "response": "This is a successful test response.", "done": True, "context": [1, 2, 3],
    "total_duration": 5000000000, "load_duration": 1000000, "prompt_eval_count": 10,
    "prompt_eval_duration": 200000000, "eval_count": 30, "eval_duration": 3000000000
}
MOCK_GEMINI_RESPONSE_SUCCESS = {
  "candidates": [{"content": {"parts": [{"text": "This is a successful Gemini test response."}],"role": "model"}, "finishReason": "STOP", "index": 0, "safetyRatings": []}],
  "promptFeedback": {"safetyRatings": []}
}
# --- VLLM ADDITION START ---
# Mock OpenAI Completion object for vLLM
class MockChoice:
    def __init__(self, text="Mock vLLM response", finish_reason="stop"):
        self.message = MagicMock()
        self.message.content = text
        self.finish_reason = finish_reason
class MockCompletion:
    def __init__(self, choices=None):
        self.choices = choices or [MockChoice()]
    def model_dump_json(self, indent=None): # Mock the Pydantic method
        return json.dumps({"choices": [{"message": {"content": c.message.content}, "finish_reason": c.finish_reason} for c in self.choices]}, indent=indent)
MOCK_VLLM_RESPONSE_SUCCESS = MockCompletion()
# --- VLLM ADDITION END ---
MOCK_GEMINI_RESPONSE_BLOCKED = {"promptFeedback": {"blockReason": "SAFETY", "safetyRatings": []}}
MOCK_GEMINI_RESPONSE_CANDIDATE_BLOCKED = {"candidates": [{"finishReason": "SAFETY", "index": 0, "safetyRatings": []}]}
MOCK_OLLAMA_MODELS_RESPONSE = {"models": [{"name": TEST_MODEL_OLLAMA.split('/')[-1], "modified_at": "...", "size": 12345},{"name": "another-model:7b", "modified_at": "...", "size": 67890}]}
MOCK_OLLAMA_PS_RESPONSE_EMPTY = {"models": []}
MOCK_OLLAMA_PS_RESPONSE_RUNNING = {"models": [{"name": "preloaded-model:latest", "size": 123, "expires_at": "..."}]}
MOCK_OLLAMA_UNLOAD_RESPONSE = {"status": "success"}

# --- Mock Default Config Data (Updated Structure) ---
MOCK_DEFAULT_FILE_CONFIG = {
    'api': {
        'gemini_api_key': 'file-gemini-key',
        'ollama_host_url': 'http://file-config-host:11434',
        'vllm_host_url': 'http://file-config-vllm:8000/v1'
        },
    'default_models': ['ollama/file-default-model:latest', TEST_MODEL_GEMINI, TEST_MODEL_VLLM],
    'code_models': ['ollama/file-code-model:latest'],
    'paths': { 'tasks_file': 'config_tasks.json', 'report_dir': './config_report', 'cache_dir': './config_cache', 'template_file': 'config_template.html' },
    'timeouts': {'request': 200, 'code_execution_per_case': 15},
    'retries': {'max_retries': 1, 'retry_delay': 3},
    'cache': {'ttl_seconds': 3600},
    'scoring': { 'ollama_perf_score': {'accuracy': 0.6, 'tokens_per_sec': 0.2, 'ram_efficiency': 0.2}, 'category_weights': {'General NLP': 0.9, 'Code Generation': 1.1, 'default': 0.6}},
    'features': { 'ram_monitor': False, 'gpu_monitor': False, 'visualizations': False, 'semantic_eval': False },
    'evaluation': {'default_min_confidence': 0.8, 'passing_score_threshold': 75.0}
}


# --- Mock Objects ---
class MockPsutilProcess:
    def __init__(self, pid, name='python', cmdline=None, rss=100 * 1024 * 1024):
        self.pid = pid; self._name = name; self._cmdline = cmdline or ['python']
        self._rss = rss; self._is_running = True
        self.info = {'pid': self.pid, 'name': self._name, 'cmdline': self._cmdline}
    def memory_info(self):
        if not self._is_running: raise psutil.NoSuchProcess(self.pid)
        mock_mem = MagicMock(); mock_mem.rss = self._rss; return mock_mem
    def is_running(self): return self._is_running
    def terminate(self): self._is_running = False

# --- Base Mock RuntimeConfig for Tests ---
class BaseMockRuntimeConfig(RuntimeConfig):
     def __init__(self, psutil_available=False, pynvml_available=False, yaml_available=True, matplotlib_available=False, sentence_transformers_available=False, openai_available=True, **kwargs): # VLLM: Add openai_available
         super().__init__()
         self.ollama_base_url = DEFAULT_OLLAMA_BASE_URL
         self.vllm_host_url = DEFAULT_VLLM_BASE_URL
         self.gemini_api_url_base = DEFAULT_GEMINI_API_URL_BASE
         self.gemini_key = "mock-test-key"
         self.request_timeout = 30; self.max_retries = 1; self.retry_delay = 1
         self.code_exec_timeout = 5; self.passing_score_threshold = 70.0; self.default_min_confidence = 0.75

         self.pyyaml_available = yaml_available; self.psutil_available = psutil_available
         self.pynvml_available = pynvml_available; self.matplotlib_available = matplotlib_available
         self.sentence_transformers_available = sentence_transformers_available
         self.openai_available = openai_available

         # Assign mock objects *if* the library is marked as available
         self.yaml = MagicMock() if yaml_available else None
         self.psutil = MagicMock() if psutil_available else None
         self.pynvml = MagicMock() if pynvml_available else None
         self.matplotlib = MagicMock() if matplotlib_available else None
         self.plt = MagicMock() if matplotlib_available else None
         self.mticker = MagicMock() if matplotlib_available else None
         self.SentenceTransformer = MagicMock() if sentence_transformers_available else None
         self.st_util = MagicMock() if sentence_transformers_available else None
         self.semantic_model = MagicMock() if sentence_transformers_available else None
         self.openai = MagicMock() if openai_available else None

         # Configure mocks if they were created
         if yaml_available: self.yaml.YAMLError = yaml.YAMLError if 'yaml' in sys.modules else Exception
         if psutil_available:
             # Ensure psutil is mocked correctly if available
             try:
                 import psutil as real_psutil
                 self.psutil.NoSuchProcess = real_psutil.NoSuchProcess
                 self.psutil.AccessDenied = real_psutil.AccessDenied
             except ImportError:
                 self.psutil.NoSuchProcess = Exception
                 self.psutil.AccessDenied = Exception
             self.psutil.pid_exists.return_value=True
             self.psutil.process_iter.return_value=[]
         if pynvml_available:
             try:
                 import pynvml as real_pynvml
                 self.pynvml.NVMLError = real_pynvml.NVMLError
             except ImportError:
                 self.pynvml.NVMLError = Exception
             self.pynvml.nvmlInit.return_value=None; self.pynvml.nvmlShutdown.return_value=None; self.pynvml.nvmlDeviceGetCount.return_value = 1
         if matplotlib_available: self.plt.subplots = MagicMock(return_value=(MagicMock(), MagicMock())) # fig, ax
         if sentence_transformers_available:
              mock_tensor = MagicMock(); mock_tensor.item.return_value = 0.8
              self.semantic_model.encode.return_value = "mock_embedding"
              self.st_util.pytorch_cos_sim.return_value = mock_tensor
        
         if openai_available:
              self.openai.OpenAI = MagicMock() # Mock the class constructor
              mock_client_instance = MagicMock() # Mock the instance returned by OpenAI()
              mock_completions = MagicMock() # Mock the completions attribute
              mock_completions.create = MagicMock(return_value=MOCK_VLLM_RESPONSE_SUCCESS) # Mock the create method
              mock_client_instance.chat.completions = mock_completions # Assign mocked completions to chat attribute
              self.openai.OpenAI.return_value = mock_client_instance # Make constructor return mock instance
              # Mock error types
              self.openai.APIError = openai.APIError if 'openai' in sys.modules else Exception
              self.openai.APITimeoutError = openai.APITimeoutError if 'openai' in sys.modules else Exception
              self.openai.APIConnectionError = openai.APIConnectionError if 'openai' in sys.modules else Exception
              self.openai.APIStatusError = openai.APIStatusError if 'openai' in sys.modules else Exception

         # Apply any other overrides passed via kwargs
         for key, value in kwargs.items():
             if hasattr(self, key): setattr(self, key, value)

# --- Test Classes ---

class TestUtils(unittest.TestCase):
    def test_format_na(self):
        self.assertEqual(utils.format_na(None), "N/A"); self.assertEqual(utils.format_na(float('nan')), "N/A")
        self.assertEqual(utils.format_na(123), "123"); self.assertEqual(utils.format_na(123.456), "123.5")
        self.assertEqual(utils.format_na(123.456, precision=2), "123.46")
        self.assertEqual(utils.format_na(123.456, suffix=" MB", precision=1), "123.5 MB")
        self.assertEqual(utils.format_na(0), "0"); self.assertEqual(utils.format_na("abc"), "abc")
    def test_truncate_text(self):
        self.assertEqual(utils.truncate_text("short text"), "short text"); self.assertEqual(utils.truncate_text(""), "")
        self.assertEqual(utils.truncate_text(None), ""); long_text = "Long text example"
        self.assertEqual(utils.truncate_text(long_text, max_len=5), "Long ...")

class TestLLMClients(unittest.TestCase):
    def setUp(self): self.mock_runtime_config = BaseMockRuntimeConfig(gemini_key="test-key", openai_available=True) # Ensure OpenAI is mocked

    # -- Ollama Tests --
    @patch('requests.post')
    def test_query_ollama_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.json.return_value = MOCK_OLLAMA_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp; expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_GENERATE_PATH}"
        text, duration, tokps, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA.split('/')[-1], "p", self.mock_runtime_config)
        mock_post.assert_called_once_with(expected_url, json=ANY, timeout=ANY); self.assertEqual(text, "This is a successful test response.")
        self.assertIsNone(error); self.assertGreater(duration, 0); self.assertAlmostEqual(tokps, 10.0, delta=0.1)
    @patch('requests.post')
    def test_query_ollama_404_error(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 404; mock_resp.json.return_value = {"error": "model not found"}
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404"); mock_post.return_value = mock_resp
        text, _, _, error = llm_clients.query_ollama("missing", "p", self.mock_runtime_config)
        self.assertEqual(text, ""); self.assertIn("404", error); self.assertIn("model not found", error)
    @patch('requests.post'); @patch('time.sleep', return_value=None)
    def test_query_ollama_500_retry_fail(self, mock_sleep, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 500; mock_resp.text = "Error"
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500"); mock_post.return_value = mock_resp
        _, _, _, error = llm_clients.query_ollama(TEST_MODEL_OLLAMA.split('/')[-1], "p", self.mock_runtime_config)
        self.assertIn("Server Error HTTP 500", error); self.assertIn("After 2 attempts", error)
        self.assertEqual(mock_post.call_count, 2); self.assertEqual(mock_sleep.call_count, 1)
    @patch('requests.get'); def test_get_local_ollama_models(self, mock_get):
        mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.json.return_value = MOCK_OLLAMA_MODELS_RESPONSE
        mock_get.return_value = mock_resp; expected_url = f"{self.mock_runtime_config.ollama_base_url}{llm_clients.OLLAMA_MODELS_PATH}"
        models = llm_clients.get_local_ollama_models(self.mock_runtime_config)
        self.assertEqual(models, {TEST_MODEL_OLLAMA.split('/')[-1], "another-model:7b"}); mock_get.assert_called_once_with(expected_url, timeout=10)
    @patch('requests.post'); def test_unload_ollama_model_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200; mock_post.return_value = mock_resp
        success = llm_clients.unload_ollama_model("unload-me", self.mock_runtime_config); self.assertTrue(success)

    # -- Gemini Tests --
    @patch('requests.post')
    def test_query_gemini_success(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_SUCCESS
        mock_post.return_value = mock_resp
        text, duration, tokps, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, "This is a successful Gemini test response."); self.assertIsNone(error); self.assertIsNone(tokps); self.assertGreater(duration, 0)
    @patch('requests.post')
    def test_query_gemini_blocked_prompt(self, mock_post):
        mock_resp = MagicMock(); mock_resp.status_code = 200; mock_resp.json.return_value = MOCK_GEMINI_RESPONSE_BLOCKED
        mock_post.return_value = mock_resp; text, _, _, error = llm_clients.query_gemini(TEST_MODEL_GEMINI, "p", self.mock_runtime_config)
        self.assertEqual(text, ""); self.assertIn("Blocked by API (Prompt): SAFETY", error)

    # --- VLLM ADDITION START ---
    # -- vLLM Tests --
    def test_query_vllm_success(self):
        # Mock is configured in BaseMockRuntimeConfig setUp
        mock_openai_class = self.mock_runtime_config.openai.OpenAI
        mock_create_method = mock_openai_class.return_value.chat.completions.create

        text, duration, tokps, error = llm_clients.query_vllm(TEST_MODEL_VLLM, "test prompt", self.mock_runtime_config)

        mock_openai_class.assert_called_once_with(base_url=self.mock_runtime_config.vllm_host_url, api_key=ANY, timeout=ANY)
        mock_create_method.assert_called_once_with(
            model=TEST_MODEL_VLLM.split('/')[-1], # Check cleaned name is used
            messages=[{"role": "user", "content": "test prompt"}],
            temperature=0.0
        )
        self.assertEqual(text, "Mock vLLM response")
        self.assertIsNone(error)
        self.assertIsNone(tokps) # Expect None for vLLM tok/s
        self.assertGreaterEqual(duration, 0) # Duration should be non-negative

    @patch('time.sleep', return_value=None) # Mock sleep for retry
    def test_query_vllm_retry_on_500(self, mock_sleep):
        mock_openai_class = self.mock_runtime_config.openai.OpenAI
        mock_create_method = mock_openai_class.return_value.chat.completions.create
        # Simulate 500 error then success
        mock_create_method.side_effect = [
            openai.APIStatusError("Server error", response=MagicMock(status_code=500), body=None),
            MOCK_VLLM_RESPONSE_SUCCESS
        ]

        text, duration, tokps, error = llm_clients.query_vllm(TEST_MODEL_VLLM, "p", self.mock_runtime_config)

        self.assertEqual(mock_create_method.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)
        self.assertEqual(text, "Mock vLLM response")
        self.assertIsNone(error)

    def test_query_vllm_fail_on_400(self):
        mock_openai_class = self.mock_runtime_config.openai.OpenAI
        mock_create_method = mock_openai_class.return_value.chat.completions.create
        mock_create_method.side_effect = openai.APIStatusError("Bad request", response=MagicMock(status_code=400), body=None)

        text, duration, tokps, error = llm_clients.query_vllm(TEST_MODEL_VLLM, "p", self.mock_runtime_config)

        self.assertEqual(mock_create_method.call_count, 1)
        self.assertEqual(text, "")
        self.assertIsNotNone(error)
        self.assertIn("HTTP 400", error)

    def test_query_vllm_no_openai_lib(self):
        config_no_openai = BaseMockRuntimeConfig(openai_available=False)
        text, duration, tokps, error = llm_clients.query_vllm(TEST_MODEL_VLLM, "p", config_no_openai)
        self.assertEqual(text, "")
        self.assertEqual(duration, 0.0)
        self.assertIsNone(tokps)
        self.assertIn("OpenAI library not available", error)
    # --- VLLM ADDITION END ---


class TestSystemMonitor(unittest.TestCase):
    def setUp(self):
        self.mock_runtime_config_psutil = BaseMockRuntimeConfig(psutil_available=True)
        self.mock_runtime_config_gpu = BaseMockRuntimeConfig(pynvml_available=True)

    def test_get_ollama_pids_found(self):
        mock_proc = MockPsutilProcess(pid=123, name='ollama', cmdline=['/usr/bin/ollama', 'serve'])
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.process_iter.return_value = [mock_proc]
        pids = system_monitor.get_ollama_pids(mock_psutil_mod, self.mock_runtime_config_psutil)
        self.assertEqual(pids, [123])

    # --- VLLM ADDITION START ---
    def test_get_vllm_pids_found(self):
        mock_proc = MockPsutilProcess(pid=456, name='python', cmdline=['python', '-m', 'vllm.entrypoints.openai.api_server', '--model', 'some/model'])
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.process_iter.return_value = [mock_proc]
        pids = system_monitor.get_vllm_pids(mock_psutil_mod, self.mock_runtime_config_psutil)
        self.assertEqual(pids, [456])

    def test_get_vllm_pids_not_found(self):
        mock_proc = MockPsutilProcess(pid=789, name='python', cmdline=['python', 'my_script.py'])
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.process_iter.return_value = [mock_proc]
        pids = system_monitor.get_vllm_pids(mock_psutil_mod, self.mock_runtime_config_psutil)
        self.assertEqual(pids, [])
    # --- VLLM ADDITION END ---

    def test_get_combined_rss(self):
        mock_proc1 = MockPsutilProcess(pid=1, rss=100 * 1024**2)
        mock_proc2 = MockPsutilProcess(pid=2, rss=200 * 1024**2)
        mock_psutil_mod = self.mock_runtime_config_psutil.psutil
        mock_psutil_mod.Process.side_effect = lambda pid: {1: mock_proc1, 2: mock_proc2}.get(pid)
        total_rss = system_monitor.get_combined_rss([1, 2], mock_psutil_mod)
        self.assertEqual(total_rss, 300 * 1024**2)

    def test_get_gpu_memory_usage_success(self):
        mock_handle = MagicMock(); mock_mem_info = MagicMock(); mock_mem_info.used = 5 * 1024**3
        mock_nvml_mod = self.mock_runtime_config_gpu.pynvml
        mock_nvml_mod.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_nvml_mod.nvmlDeviceGetMemoryInfo.return_value = mock_mem_info
        mem_used = system_monitor.get_gpu_memory_usage(mock_nvml_mod, 0); self.assertEqual(mem_used, 5 * 1024**3)

class TestEvaluation(unittest.TestCase):
    def setUp(self): self.runtime_config = BaseMockRuntimeConfig(yaml_available=True, sentence_transformers_available=True)
    def test_evaluate_response_dispatch_code(self):
        task = {"type": "code_generation", "function_name": "f", "test_cases": []}
        with patch('evaluation.evaluator.execute_and_test_code') as mock_exec:
            mock_exec.return_value = (True, "OK"); metric, details = evaluation.evaluate_response(task, "code", self.runtime_config)
            mock_exec.assert_called_once(); self.assertTrue(metric); self.assertEqual(details, "OK")
    def test_evaluate_response_dispatch_semantic(self):
        task = {"type": "summarization", "evaluation_method": "semantic", "expected_keywords": ["ref"]}
        with patch('evaluation.evaluator.evaluate_semantic_similarity') as mock_sem:
            mock_sem.return_value = (95.0, "High"); metric, details = evaluation.evaluate_response(task, "response", self.runtime_config)
            mock_sem.assert_called_once(); self.assertEqual(metric, 95.0); self.assertEqual(details, "High")

class TestScoring(unittest.TestCase):
    def test_compute_scores_with_vllm(self):
        mock_results = {
            "ollama_model": {"_summary": {"model_name": "ollama_model", "provider": "ollama", "status": "Completed", "accuracy": 90.0, "tokens_per_sec_avg": 100.0, "delta_ram_mb": 500.0, "per_type": {"T1": {"accuracy": 90.0, "count": 1, "success_api_calls": 1}}}},
            "vllm_model": {"_summary": {"model_name": "vllm_model", "provider": "vllm", "status": "Completed", "accuracy": 85.0, "delta_ram_mb": 400.0, "per_type": {"T1": {"accuracy": 85.0, "count": 1, "success_api_calls": 1}}}},
            "gemini_model": {"_summary": {"model_name": "gemini_model", "provider": "gemini", "status": "Completed", "accuracy": 80.0, "per_type": {"T1": {"accuracy": 80.0, "count": 1, "success_api_calls": 1}}}},
            "_task_categories": {"C1": ["task1"]}, "_task_definitions": {"task1": {"type": "T1"}},
            "_ollama_score_weights_used": {"accuracy": 0.5, "tokens_per_sec": 0.3, "ram_efficiency": 0.2}
        }
        category_weights = {"C1": 1.0, "default": 1.0}; default_weight = 1.0
        scoring.compute_performance_scores(mock_results, category_weights, default_weight)
        # Check Ollama score calculation
        self.assertIsNotNone(mock_results["ollama_model"]["_summary"]["ollama_perf_score"])
        self.assertGreater(mock_results["ollama_model"]["_summary"]["ollama_perf_score"], 0)
        # Check vLLM score (Ollama score should be None)
        self.assertIsNone(mock_results["vllm_model"]["_summary"]["ollama_perf_score"])
        # Check Overall Weighted Score
        self.assertAlmostEqual(mock_results["ollama_model"]["_summary"]["overall_weighted_score"], 90.0)
        self.assertAlmostEqual(mock_results["vllm_model"]["_summary"]["overall_weighted_score"], 85.0)
        self.assertAlmostEqual(mock_results["gemini_model"]["_summary"]["overall_weighted_score"], 80.0)

class TestCacheManager(unittest.TestCase):
    def setUp(self): self.temp_dir=tempfile.mkdtemp(); self.cache_dir=pathlib.Path(self.temp_dir); safe_name=re.sub(r"[^a-zA-Z0-9_\-]+", "_", TEST_BENCHMARK_NAME.lower()); self.cache_file=self.cache_dir / f"cache_{safe_name}.json"; self.cache_ttl=3600
    def tearDown(self): shutil.rmtree(self.temp_dir)
    def test_save_load_cache(self): data={"m1": {"_summary": {"acc": 80.0}}}; cache_manager.save_cache(self.cache_file, data); loaded=cache_manager.load_cache(self.cache_file, self.cache_ttl); self.assertEqual(loaded, data)

class TestReporting(unittest.TestCase):
    def setUp(self):
        self.temp_dir_obj=tempfile.TemporaryDirectory(); self.temp_dir=self.temp_dir_obj.name
        self.runtime_config=BaseMockRuntimeConfig(matplotlib_available=True, visualizations_enabled=True, openai_available=True)
        self.runtime_config.report_dir=pathlib.Path(self.temp_dir); self.runtime_config.report_img_dir=self.runtime_config.report_dir/"images"; self.runtime_config.report_img_dir.mkdir(parents=True, exist_ok=True); self.runtime_config.html_template_file=pathlib.Path("./report_template.html")
        self.results={"ollama_a":{"_summary":{"status":"Completed","model_name":"ollama_a","provider":"ollama","accuracy":90.0,"overall_weighted_score":90.0, "ollama_perf_score": 80.0, "tokens_per_sec_avg": 100.0, "delta_ram_mb": 500}}, "vllm_b":{"_summary":{"status":"Completed","model_name":"vllm_b","provider":"vllm","accuracy":80.0,"overall_weighted_score":80.0, "delta_ram_mb": 400}}, "gem_c":{"_summary":{"status":"Completed","model_name":"gem_c","provider":"gemini","accuracy":70.0,"overall_weighted_score":70.0}}, "_t_def":{},"_t_cat":{},"_cat_w":{}}
        if self.runtime_config.matplotlib_available: self.runtime_config.plt.savefig=MagicMock(); self.runtime_config.plt.subplots=MagicMock(return_value=(MagicMock(),MagicMock())); self.runtime_config.plt.close=MagicMock()
    def tearDown(self): self.temp_dir_obj.cleanup()
    def test_generate_ranking_plot_skips_none(self):
        if not self.runtime_config.matplotlib_available: self.skipTest("Matplotlib mock unavailable")
        # Test plotting tok/s (should only include ollama_a)
        path=reporting.generate_ranking_plot(self.results,"tokens_per_sec_avg","TokPS Plot","TokPS","tokps.png",self.runtime_config)
        self.assertEqual(path,"images/tokps.png")
        # Check that plt.barh was called with only 1 model's data
        call_args, _ = self.runtime_config.plt.subplots.return_value[1].barh.call_args
        models_plotted = call_args[0]
        self.assertEqual(len(models_plotted), 1)
        self.assertEqual(models_plotted[0], "ollama_a")
    def test_generate_html_report_handles_providers(self):
        html = reporting.generate_html_report(self.results, "Test Report", {}, self.runtime_config)
        self.assertIn("<td>OLLAMA</td>", html) # Check provider column
        self.assertIn("<td>VLLM</td>", html)
        self.assertIn("<td>GEMINI</td>", html)
        self.assertIn(">80.0</td>", html) # Ollama perf score
        self.assertIn(">N/A</td>", html) # Placeholder for vLLM/Gemini Ollama score
        self.assertIn(">100.0</td>", html) # Ollama Tok/s
        self.assertIn(">N/A</td>", html) # Placeholder for vLLM/Gemini Tok/s
        self.assertIn("500 MB</td>", html) # Ollama RAM delta
        self.assertIn("400 MB</td>", html) # vLLM RAM delta
        self.assertIn(">N/A</td>", html) # Placeholder for Gemini RAM delta

class TestCLIConfigLoading(unittest.TestCase):
    def setUp(self):
        self.load_config_patcher=patch('benchmark_cli.load_config_from_file', return_value=MOCK_DEFAULT_FILE_CONFIG); self.mock_load_config=self.load_config_patcher.start()
        self.setup_config_patcher=patch('benchmark_cli.setup_runtime_config'); self.mock_setup_config=self.setup_config_patcher.start(); self.mock_setup_config.return_value=BaseMockRuntimeConfig(openai_available=True) # Ensure mock config has openai
        self.run_benchmark_patcher=patch('benchmark_cli.run_benchmark_set', return_value={"m1":{"_summary":{"status":"Completed"}}}); self.mock_run_benchmark=self.run_benchmark_patcher.start()
        self.report_gen_patcher=patch('benchmark_cli.generate_html_report', return_value="<html></html>"); self.mock_report_gen=self.report_gen_patcher.start()
        self.report_save_patcher=patch('benchmark_cli.save_report', return_value=pathlib.Path(".")); self.mock_report_save=self.report_save_patcher.start()
        self.mkdir_patcher=patch('pathlib.Path.mkdir'); self.mock_mkdir=self.mkdir_patcher.start()
        self.load_tasks_patcher=patch('benchmark_cli.load_and_validate_tasks', return_value=({"t1":{}},{"C1":["t1"]})); self.mock_load_tasks=self.load_tasks_patcher.start()
        self.check_dep_patcher=patch('benchmark_cli.check_dependencies'); self.mock_check_dep=self.check_dep_patcher.start()
        self.get_local_models_patcher=patch('llm_clients.get_local_ollama_models', return_value={'file-default-model:latest'}); self.mock_get_local_models=self.get_local_models_patcher.start()
        self.get_running_models_patcher=patch('llm_clients.get_ollama_running_models', return_value=[]); self.mock_get_running_models=self.get_running_models_patcher.start()
        self.unload_model_patcher=patch('llm_clients.unload_ollama_model', return_value=True); self.mock_unload_model=self.unload_model_patcher.start()
    def tearDown(self): patch.stopall()
    def test_cli_uses_config_file_defaults_with_vllm(self):
        test_args=['benchmark_cli.py']; self.setup_config_patcher.stop() # Need real setup
        with patch.object(sys,'argv',test_args): benchmark_cli.main()
        self.mock_load_config.assert_called_once_with(DEFAULT_TEST_CONFIG_PATH); self.mock_run_benchmark.assert_called_once()
        call_kwargs=self.mock_run_benchmark.call_args; final_config=call_kwargs.get('runtime_config')
        # Check values loaded from MOCK_DEFAULT_FILE_CONFIG
        self.assertEqual(final_config.gemini_key,'file-gemini-key')
        self.assertEqual(final_config.ollama_base_url,'http://file-config-host:11434')
        self.assertEqual(final_config.vllm_host_url,'http://file-config-vllm:8000/v1') # VLLM Check
        self.assertEqual(final_config.models_to_benchmark,['ollama/file-default-model:latest', TEST_MODEL_GEMINI, TEST_MODEL_VLLM]) # VLLM Check
        self.assertEqual(final_config.code_models_to_benchmark,['ollama/file-code-model:latest'])
        self.assertFalse(final_config.ram_monitor_enabled)
        self.setup_config_patcher.start() # Restart patcher for other tests

    @patch.dict(os.environ,{"GEMINI_API_KEY":"env-key","OLLAMA_HOST":"http://env-host:11434", "VLLM_HOST": "http://env-vllm:9000/v1"},clear=True)
    def test_cli_overrides_env_overrides_file(self):
        test_args=['benchmark_cli.py','--gemini-key','cli-key','--test-model','ollama/cli-model', '--vllm-host', 'http://cli-vllm:7000/v1']; self.setup_config_patcher.stop()
        with patch.object(sys,'argv',test_args): benchmark_cli.main()
        self.mock_run_benchmark.assert_called_once(); _, call_kwargs=self.mock_run_benchmark.call_args; final_config=call_kwargs.get('runtime_config')
        self.assertEqual(final_config.gemini_key,'cli-key') # CLI > ENV > File
        self.assertEqual(final_config.ollama_base_url,'http://env-host:11434') # ENV > File
        self.assertEqual(final_config.vllm_host_url,'http://cli-vllm:7000/v1') # CLI > ENV > File - VLLM Check
        self.assertEqual(final_config.models_to_benchmark,['ollama/cli-model']) # CLI overrides File default_models
        self.assertEqual(final_config.code_models_to_benchmark,['ollama/cli-model']) # CLI test_model applies to code if no --code-model
        self.setup_config_patcher.start()

    def test_get_provider_from_model_name(self):
        self.assertEqual(get_provider_from_model_name("llama3:8b"), "ollama")
        self.assertEqual(get_provider_from_model_name("ollama/llama3:8b"), "ollama")
        self.assertEqual(get_provider_from_model_name("vllm/model-id"), "vllm")
        self.assertEqual(get_provider_from_model_name("vllm/namespace/model-id"), "vllm")
        self.assertEqual(get_provider_from_model_name("gemini-1.5-flash"), "gemini")
        self.assertEqual(get_provider_from_model_name("models/gemini-pro"), "gemini")
        self.assertEqual(get_provider_from_model_name("some-other-model"), "ollama") # Default case

# Main entry point for tests
if __name__ == '__main__':
    # Ensure psutil is importable if tests require it, even if not installed globally
    # This helps the mock setup work correctly.
    try:
        import psutil
    except ImportError:
        sys.modules['psutil'] = MagicMock() # Create a mock module if not installed
    try:
        import pynvml
    except ImportError:
        sys.modules['pynvml'] = MagicMock()
    try:
        import matplotlib
        import matplotlib.pyplot
        import matplotlib.ticker
    except ImportError:
         sys.modules['matplotlib'] = MagicMock()
         sys.modules['matplotlib.pyplot'] = MagicMock()
         sys.modules['matplotlib.ticker'] = MagicMock()
    try:
        import sentence_transformers
    except ImportError:
        sys.modules['sentence_transformers'] = MagicMock()

    unittest.main()
