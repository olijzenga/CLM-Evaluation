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

import torch
from transformers import pipeline

from clm.model import MaskPredictModel
from clm.model.mask_predict import (
    MaskPredictResult,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
    MaskPredictModelVariant,
)


class GPTNeo:
    def __init__(self, load_config: ModelLoadConfig, model_name: str = "eleutherai/gpt-neo-125m") -> None:
        self.pipeline = pipeline("text-generation", model=model_name, **load_config.get_load_parameters())

    def complete(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        results = list(
            self.pipeline(
                text,
                do_sample=True,
                return_scores=True,
                **sampling_config.get_sampling_parameters(),
            )
        )
        print(results)
        return [MaskPredictResult(result["sequence"], None, result["score"]) for result in results]


class GPTNeoMaskPredictModel(MaskPredictModel):
    NAME = "gptneo"
    PARAMETER_DATATYPE = ParameterDataType.F32

    VARIANTS = [
        MaskPredictModelVariant("125M", default_beam_size=10),
        MaskPredictModelVariant("1.3B", default_beam_size=10),
        MaskPredictModelVariant("20B", default_beam_size=10)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.model = GPTNeo(self.load_config, f"eleutherai/gpt-neo-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.model.complete(text, sampling_config)

    def get_mask(self, mask: str) -> str:
        return ""

    def get_does_multi_token_prediction(self) -> bool:
        return True
