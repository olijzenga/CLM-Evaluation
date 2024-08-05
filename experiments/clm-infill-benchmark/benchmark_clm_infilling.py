# Copyright (c) 2024 Oebele Lijzenga
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import logging
import os
import shutil
import subprocess
import time
import numpy as np
import random
import fire
import torch

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import Thread


from clm.mask_predict import MaskPredictor
from clm.model import (
    MaskPredictModel,
    ModelSamplingPreferences,
    ModelSamplingConfig,
    QuantizationMode,
    ModelLoadPreferences,
)
from clm.clms import MaskPredictModelFactory, ALL_MODELS_BY_NAME
from clm.setup import setup_logging

log = logging.getLogger(__name__)

BENCHMARK_SOURCE_DIR = "humaneval-java-infill"

BATCH_TEST_TIMEOUT = 120.0
SINGLE_TEST_TIMEOUT = 20.0
GPU_MEMORY_USAGE_POLL_INTERVAL = 1.0


def get_buggy_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "src", "main", "java", "humaneval", "buggy")


def get_test_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "src", "test", "java", "humaneval")


def load_buggy_files(base_dir: str) -> dict[str, str]:
    """Returns tuples of file name and content of benchmark samples"""
    file_dir = get_buggy_dir(base_dir)
    file_names = os.listdir(file_dir)

    result = {}
    for file_name in file_names:
        with open(os.path.join(file_dir, file_name), "r") as f:
            result[file_name] = f.read()

    return result


def exec(
    command: str, log_file: str | None, cwd: str | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess:
    def _log(stdout, stderr, returncode):
        if log_file is not None:
            with open(log_file, "wb") as f:
                f.write(b"command:\n")
                f.write((command + "\n").encode())
                f.write(b"stdout:\n")
                f.write(stdout)
                f.write(b"\n")
                f.write(b"sterr:\n")
                f.write(stderr)
                f.write(b"\n")
                f.write(f"exit code: {returncode}\n".encode())

    try:
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout or 30,
        )
    except subprocess.TimeoutExpired as e:
        _log(e.stdout or bytes(), e.stderr or bytes(), str(e))
        raise

    _log(result.stdout, result.stderr, result.returncode)

    return result


def remove_non_ascii(text: str, replacement: str = "[UNK]") -> str:
    return "".join([i if ord(i) < 128 else replacement for i in text])


def get_visible_gpus() -> list[int]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [int(i) for i in cuda_visible_devices.split(",")]
    else:
        return [0]


def get_gpu_memory_usage_mib() -> int | None:
    memory_usage = 0
    for gpu_index in get_visible_gpus():
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(gpu_index),
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
            timeout=10.0
        )
        memory_usage += int(result.stdout.strip())
    return memory_usage


class BatchTestException(Exception):
    """ Raised when batch test execution failed, and individual test execution should be used instead """
    pass


class ClmInfillingBenchmarkSuite:
    def __init__(
        self,
        model: MaskPredictModel | None,
        workdir: str,
        sampling_config: ModelSamplingConfig,
        javac_executable: str,
        mvn_executable: str,
        n: int,
        metadata: dict,
        fast_testing: bool,
        nr_cores: int,
        cuda: bool,
    ):
        self._model: MaskPredictModel | None = model
        self._workdir: str = workdir
        self._sampling_config: ModelSamplingConfig = sampling_config
        self._javac_executable: str = javac_executable
        self._mvn_executable: str = mvn_executable
        self._n: int = n
        self._metadata: dict = metadata
        self._fast_testing: bool = fast_testing
        self._nr_cores: int = nr_cores
        self._result_dirs: list[str] = [
            os.path.join(workdir, f"result_{n}") for n in range(self._sampling_config.num_return_sequences)
        ]
        self._buggy_files: dict[str, str] = load_buggy_files(BENCHMARK_SOURCE_DIR)
        self._total_mask_predict_time: float = 0.0
        self._cuda: bool = cuda

        self._max_memory_usage_mib: int = 0
        self._memory_usage_thread: Thread | None = None
        self._infill_running: bool = False

    def _max_memory_usage_worker(self) -> None:
        while self._infill_running:
            try:
                memory_usage = get_gpu_memory_usage_mib()
            except subprocess.CalledProcessError as e:
                log.error("Failed to retrieve GPU memory usage", exc_info=e)
                self._max_memory_usage_mib = -1
                return

            if memory_usage > self._max_memory_usage_mib:
                self._max_memory_usage_mib = memory_usage

            time.sleep(GPU_MEMORY_USAGE_POLL_INTERVAL)

    def _run_infilling(self) -> None:
        log.info("Start of infilling")
        for i, (file_name, content) in enumerate(self._buggy_files.items()):
            log.info(f"Running infill for {file_name} {i+1}/{len(self._buggy_files)}")

            mask_predictor = MaskPredictor(content, self._model, self._sampling_config)

            start_time = time.time()
            results = mask_predictor.predict()
            self._total_mask_predict_time += time.time() - start_time

            for j, result_dir in enumerate(self._result_dirs):
                dst_file = os.path.join(get_buggy_dir(result_dir), file_name)

                if j >= len(results):
                    # No result, move to noresult file
                    full_result = "no result"
                    dst_file += ".noresult"
                else:
                    # Write full result with mask replacement to comment at the end of the file
                    result = results[j]
                    full_result = result.full_result
                    full_result += "\n//Mask replacement:\n"
                    for line in result.mask_replacements[0].replacement.split("\n"):
                        full_result += f"//{line}\n"

                    # Remove non-ascii characters
                    full_result = remove_non_ascii(full_result)

                with open(dst_file, "w") as f:
                    f.write(full_result)
        log.info("Infilling completed")

    def _compile(self, file_path: str, target_dir: str, classpath: str = "") -> bool:
        classpath_arg = "" if classpath == "" else f"-cp {classpath}"
        command = f"{self._javac_executable} -d {target_dir} {classpath_arg} {file_path}"
        result = exec(command, file_path.replace(".java", "_compile_log.txt"))
        return result.returncode == 0

    def _test(
        self, test_file_name: str, result_dir: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess:
        if test_file_name.endswith(".java"):
            test_file_name = test_file_name[:-5]

        command = f"{self._mvn_executable} test -Dtest={test_file_name}"
        result = exec(
            command,
            os.path.join(get_test_dir(result_dir), test_file_name.replace("*", "ALL")) + "_log.txt",
            cwd=result_dir,
            timeout=timeout,
        )

        return result

    def _compile_single_result(self, result_dir: str, file_name: str) -> str | None:
        target_dir = os.path.join(result_dir, "target")

        fixed_file_path = os.path.join(get_buggy_dir(result_dir), file_name)
        test_file_name = "TEST_" + file_name
        test_file_path = os.path.join(get_test_dir(result_dir), test_file_name)

        if not os.path.isfile(fixed_file_path):
            return "NO_RESULT"

        compile_success = self._compile(fixed_file_path, target_dir)
        if not compile_success:
            return "COMPILE_FAIL"

        compile_success = self._compile(
            test_file_path,
            target_dir,
            classpath=f"{os.path.join(result_dir, 'lib/junit4-4.12.jar')}:{os.path.join(result_dir, 'src/main/java')}",
        )
        if not compile_success:
            return "TEST_COMPILE_FAIL"

        return None

    def _test_single_result(self, result_dir: str, file_name: str) -> str:
        test_file_name = "TEST_" + file_name

        try:
            test_result = self._test(test_file_name, result_dir, timeout=SINGLE_TEST_TIMEOUT)
        except subprocess.TimeoutExpired:
            return "TEST_TIMEOUT"

        if test_result.returncode == 0:
            return "SUCCESS"
        else:
            return "TEST_FAIL"

    def _test_single_result_dir_separately(self, result_dir: str, compile_fails: list[str]) -> dict:
        """Evaluates a result folder by executing each test separately."""
        result_nr = int(result_dir.split("_")[-1])
        buggy_file_names = [
            file_name for file_name in self._buggy_files.keys() if file_name not in compile_fails
        ]
        results = {}
        for i, file_name in enumerate(buggy_file_names):
            i = buggy_file_names.index(file_name)
            log.info(f"Executing tests for {file_name} {i+1}/{len(buggy_file_names)} of result {result_nr}")
            results[file_name] = self._test_single_result(result_dir, file_name)

        return results

    def _test_single_result_dir_batch(self, result_dir: str, compile_fails: list[str]) -> dict:
        """Evaluates a result folder by executing all tests at once."""
        result_nr = int(result_dir.split("_")[-1])

        if len(compile_fails) == len(self._buggy_files.keys()):
            log.info(f"Skipping tests for result {result_nr} as there are no compileable results")
            return {}

        results = {
            file_name: "SUCCESS" for file_name in self._buggy_files.keys() if file_name not in compile_fails
        }

        log.info(f"Executing tests for result {result_nr} in batch")
        try:
            test_result = self._test("TEST_*", result_dir, timeout=BATCH_TEST_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            raise BatchTestException(str(e))

        # Remove non-ascii bytes
        text = bytes([b for b in test_result.stdout if 0 <= b <= 127]).decode()

        missing_results = [
            file_name for file_name in results.keys() if ("Running humaneval.TEST_" + file_name.replace('.java', '')) not in text
        ]
        if len(missing_results) != 0:
            for missing_result in missing_results:
                log.debug(f"Missing result in batch test for file {missing_result}")
            raise BatchTestException(f"{len(missing_results)} tests were not executed")

        # Split off part that contains the result summary
        text = text.split("[INFO] Results:", 1)[1]
        assert "[INFO] Results:" not in text

        # Find failing tests from result
        for file_name in results.keys():
            if f"[ERROR]   TEST_{file_name.replace('.java', '')}." in text:
                results[file_name] = "TEST_FAIL"

        return results

    def _test_single_result_dir(self, result_dir: str, compile_fails: list[str]) -> dict:
        if self._fast_testing:
            return self._test_single_result_dir_batch(result_dir, compile_fails)
        else:
            return self._test_single_result_dir_separately(result_dir, compile_fails)

    def _compile_result_dir(self, result_dir: str) -> dict:
        result_nr = int(result_dir.split("_")[-1])
        buggy_file_names = list(self._buggy_files.keys())

        target_dir = os.path.join(result_dir, "target")
        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        def _compile(file_name: str) -> str | None:
            j = buggy_file_names.index(file_name)
            log.info(f"Compiling files for {file_name} {j+1}/{len(self._buggy_files)} of result {result_nr}")
            return self._compile_single_result(result_dir, file_name)

        with ThreadPoolExecutor(max_workers=int(self._nr_cores / len(self._result_dirs))) as executor:
            results = list(executor.map(_compile, buggy_file_names))

        compile_results = {}
        for file_name, result in zip(buggy_file_names, results):
            if result is not None:
                compile_results[file_name] = result

                source_file_path = os.path.join(get_buggy_dir(result_dir), file_name)
                if os.path.isfile(source_file_path):
                    shutil.move(source_file_path, source_file_path + ".fail")
                test_file_path = os.path.join(get_test_dir(result_dir), "TEST_" + file_name)
                shutil.move(test_file_path, test_file_path + ".fail")

        return compile_results

    def evaluate_results(self) -> dict:
        log.info("Evaluating results")

        # Evaluate results for each result directory in a separate thread
        with ThreadPoolExecutor(max_workers=len(self._result_dirs)) as executor:
            compile_results = list(executor.map(self._compile_result_dir, self._result_dirs))

        def _test_loop():
            compile_and_test_results = []
            for result_dir, compile_result in zip(self._result_dirs, compile_results):
                try:
                    test_result = self._test_single_result_dir(result_dir, list(compile_result.keys()))
                except BatchTestException as e:
                    log.warning(f"Fast testing failed with '{e}', falling back to slow testing")
                    self._fast_testing = False
                    return _test_loop()  # Redo entire loop

                compile_and_test_results.append({**compile_result, **test_result})
            return compile_and_test_results

        # Combine results
        results = {file_name: [] for file_name in self._buggy_files.keys()}
        for compile_and_test_result in _test_loop():
            for file_name, result in compile_and_test_result.items():
                results[file_name].append(result)

        return results

    def run(self) -> None:
        log.info("Creating result directories")

        skipped_files = sorted(list(self._buggy_files.keys()))[self._n :]
        for result_dir in self._result_dirs:
            shutil.copytree(BENCHMARK_SOURCE_DIR, result_dir)
            # clear the 'buggy' directory
            buggy_dir = get_buggy_dir(result_dir)
            shutil.rmtree(buggy_dir)
            os.mkdir(buggy_dir)

            # Remove ignored buggy files
            for file_name in skipped_files:
                test_file_path = os.path.join(get_test_dir(result_dir), "TEST_" + file_name)
                os.remove(test_file_path)

        # Remove data for ignored buggy instances
        for file_name in skipped_files:
            del self._buggy_files[file_name]

        start_time = time.time()

        self._infill_running = True
        if self._cuda:
            self._memory_usage_thread = Thread(target=self._max_memory_usage_worker)
            self._memory_usage_thread.start()

        try:
            self._run_infilling()
        finally:
            self._infill_running = False
            if self._cuda:
                self._memory_usage_thread.join(GPU_MEMORY_USAGE_POLL_INTERVAL * 5)
                if self._memory_usage_thread.is_alive():
                    raise Exception("Failed to join memory usage thread")

        results = self.evaluate_results()

        total_time = time.time() - start_time

        # Write results to single file
        file_content = {
            "config": {
                "model_name": self._model.model_name if self._model else None,
                "model_variant": self._model.model_variant.name if self._model else None,
                "n": self._n,
                "sampling_config": self._sampling_config.__dict__,
                "workdir": self._workdir,
                "completed_at": datetime.now().timestamp(),
                "execution_time": total_time,
                "mask_predict_time": self._total_mask_predict_time,
                "fast_testing": self._fast_testing,
                "nr_cores": self._nr_cores,
                "max_memory_usage_mib": self._max_memory_usage_mib,
                **self._metadata,
            },
            "results": results,
        }
        with open(os.path.join(self._workdir, "results.json"), "w") as f:
            f.write(json.dumps(file_content, indent=4))


def main(
    model_name: str,
    model_variant: str | None = None,
    nr_results: int = 5,
    num_beams: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    javac_executable: str | None = "javac",
    mvn_executable: str | None = "mvn",
    device: str = "cuda",
    quantization_mode: str | None = None,
    n: int = 999,
    seed: int = 0,
    debug: bool = False,
    workdir_suffix: str = "",
    fast_testing: bool = True,
    nr_cores: int | None = None,
    workdir: str | None = None
):
    """
    Args:
        model_name: Name of the CLM. See clm/clms for options.
        model_variant: Model variant / size. See clm/clms for options per CLM.
        nr_results: Number of infills generated for each infill task.
        num_beams: Number of beams for beam search.
        top_p: Top P for nucleus sampling.
        temperature: Temperature for nucleus sampling.
        javac_executable: Path to the javac executable. Explicitly provide the location of javac if its not on your PATH.
        mvn_executable: Path to the mvn executable. Explicitly provide the location of mvn if its not on your PATH.
        device: Device on which the CLM is run. Use 'cuda' for GPU, and 'cpu' for CPU.
        quantization_mode: Mode used to quantize the CLM. One of (4bit, 8bit, 16bit, 32bit). Quantization is not applied if this parameter is not set explicitly.
        n: The first n infills generated by each CLM are considered.
        seed: Seed for sampling of CLMs. Can be used for reproducibility.
        debug: Whether to use debug logging.
        workdir_suffix: Suffix to add to the auto-generated name of the output directory.
        fast_testing: Whether to enable batch testing, where all tests are run at once.
        nr_cores: Number of cores used to execute tests in parallel. This setting is only used when fast testing is NOT used.
        workdir: Name of the output directory. By default, a name is automatically generated.

    Returns:

    """

    setup_logging(logging.DEBUG if debug else logging.INFO)

    log.info("Running benchmark_clm_infilling.py with the following settings")
    for k, v in locals().items():
        log.info(f"{str(k).ljust(20)}= {v}")

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None and "cuda" in device:
        log.warning("CUDA_VISIBLE_DEVICES is not set, only capturing memory usage of GPU 0")

    if quantization_mode is not None:
        quantization_mode = QuantizationMode.from_string(quantization_mode)

    if nr_cores is None:
        nr_cores = nr_results
    else:
        if not (nr_cores / nr_results).is_integer():
            raise ValueError(f"nr_cores must be a multiple of nr_results")

    load_preferences = ModelLoadPreferences(quantization_mode, device)
    sampling_preferences = ModelSamplingPreferences(
        num_return_sequences=nr_results,
        do_beam_search=top_p is None and temperature is None,
        num_beams=num_beams,
        top_p=top_p,
        temperature=temperature,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    model = MaskPredictModelFactory(load_preferences).get_model(model_name, model_variant)
    sampling_config = ModelSamplingConfig.create_for_preferences_and_model(sampling_preferences, model)

    if workdir is None:
        dirname = f"{model_name.lower()}_{model.model_variant.name.lower()}_" + datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )
        if n != 999:
            dirname += f"_n_{n}"
        if workdir_suffix != "":
            dirname += "_" + workdir_suffix
        workdir = os.path.join("workdirs", dirname)
        os.mkdir(workdir)

    log.info("Workdir is " + workdir)

    # Test availability of external binaries
    exec(f"{javac_executable} -version", os.path.join(workdir, "javac_sanity.log")).check_returncode()
    exec(f"{mvn_executable} --version", os.path.join(workdir, "mvn_sanity.log")).check_returncode()

    metadata = {
        "load_preferences": load_preferences.to_json(),
        "sampling_preferences": sampling_preferences.__dict__,
        "load_config": model.load_config.to_json() if model else None,
        "seed": seed,
    }
    suite = ClmInfillingBenchmarkSuite(
        model,
        workdir,
        sampling_config,
        javac_executable,
        mvn_executable,
        n,
        metadata,
        fast_testing,
        nr_cores,
        "cuda" in device,
    )
    suite.run()

    log.info("Done")
    log.info(f"Results stored in {workdir}")


if __name__ == "__main__":
    fire.Fire(main)
