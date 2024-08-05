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
from transformers import AutoModel, AutoTokenizer

from clm.model import (
    MaskPredictModel,
    MaskPredictResult,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant


class CodeGeeX2:
    def __init__(self, load_config: ModelLoadConfig, model_variant="6b") -> None:
        model_name = f"THUDM/codegeex2-{model_variant}"
        self.device = load_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **load_config.get_load_parameters()
        )

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer.encode(text, return_tensorts="pt").to(self.model.device)
        outputs = self.model.generate(input_ids, **sampling_config.get_sampling_parameters())

        return [MaskPredictResult(output, None, -1) for output in outputs]


class CodeGeeX2MaskPredictModel(MaskPredictModel):
    NAME = "codegeex2"
    PARAMETER_DATATYPE = ParameterDataType.BF16

    VARIANTS = [
        MaskPredictModelVariant("6B", default_beam_size=10)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codegeex2 = CodeGeeX2(self.load_config, self.model_variant.name)

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codegeex2.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return ""
