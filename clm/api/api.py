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

import time
import flask
import os
import torch
import subprocess

from threading import Thread

seed = int(os.getenv('CLM_SEED', '0'))
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

from flask import Flask, request, jsonify
from marshmallow import ValidationError

from clm.api.dto import MaskPredictRequestDTO
from clm.mask_predict import MaskPredictor
from clm.model import ModelSamplingConfig
from clm.model.mask_predict import ModelSamplingPreferences, ModelLoadPreferences, QuantizationMode
from clm.clms import MaskPredictModelFactory
from clm.setup import setup_logging

setup_logging()
app = Flask(__name__)

device_name = os.getenv("CUDA_DEVICE", "cuda")
quantization_mode = QuantizationMode.from_string(os.getenv("CLM_QUANTIZATION_MODE")) if os.getenv("CLM_QUANTIZATION_MODE") else None
model_factory = MaskPredictModelFactory(ModelLoadPreferences(device_name=device_name, quantization_mode=quantization_mode))

print("IMPORTANT: Using device", device_name)
if quantization_mode:
    print("IMPORTANT: Using quantization mode", quantization_mode)

verbose = bool(os.getenv('CLM_API_VERBOSE', False))
if verbose:
    print("Verbose mode is enabled")

def get_visible_gpus() -> list[int]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return [int(i) for i in cuda_visible_devices.split(",")]
    else:
        return [0]

def get_gpu_memory_usage_mib() -> int:
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

class VRAMUsage:
    def __init__(self):
        self._usage: int = 0
        self._stop: bool = False
        self._thread: Thread = Thread(target=self._loop)
        self._measure()

    def _measure(self):
        usage = get_gpu_memory_usage_mib()
        if usage > self._usage:
            self._usage = usage

    def _loop(self):
        while not self._stop:
            self._measure()
            time.sleep(0.25)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join()

    def get(self) -> int:
        return self._usage

@app.route("/mask_predict", methods=["POST"])
def mask_predict():
    try:
        parameters = MaskPredictRequestDTO().load(request.json)
    except ValidationError as err:
        return jsonify(err.messages), 400

    start_time = time.time()
    model = model_factory.get_model(parameters["model_name"], parameters["model_variant"])
    model_load_time = time.time() - start_time

    sampling_preferences = ModelSamplingPreferences(num_return_sequences=parameters["nr_results"])
    sampling_config = ModelSamplingConfig.create_for_preferences_and_model(sampling_preferences, model)

    if verbose:
        print("\n\n\nPrompt:")
        print(parameters["text"])

    predictor = MaskPredictor(parameters["text"], model, sampling_config)
    vram_usage = VRAMUsage()
    vram_usage.start()

    start_time = time.time()
    results = predictor.predict()
    predict_time = time.time() - start_time

    vram_usage.stop()

    # if verbose:
    print("Infill Result:")
    print(results[0].mask_replacements[0].replacement)

    print(f"Max VRAM MiB was {vram_usage.get()}")
    print(f"Generated infill in {round(predict_time, 3)} seconds")

    return flask.jsonify(
        {
            "results": results,
            "model_load_time": round(model_load_time, 1),
            "predict_time": round(predict_time, 1),
            'max_vram_mib': vram_usage.get()
        }
    )

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "OK"
