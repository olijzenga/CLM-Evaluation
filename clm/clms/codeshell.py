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
from transformers import AutoTokenizer, AutoModelForCausalLM

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant
from clm.util import remove_prefix_ignoring_whitespaces


class CodeShell:
    def __init__(self, load_config: ModelLoadConfig, model_name="wisdomshell/codeshell-7B"):
        self.device: torch.device = load_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **load_config.get_load_parameters()
        )

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        text = f"<fim_prefix>{text}<fim_middle>"

        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            input_ids, pad_token_id=self.tokenizer.eos_token_id, **sampling_config.get_sampling_parameters()
        )

        plain_input = text.replace("<fim_prefix>", "").replace("<fim_suffix>", "").replace("<fim_middle>", "")

        return [
            MaskPredictResult(
                None,
                [
                    remove_prefix_ignoring_whitespaces(
                        plain_input, self.tokenizer.decode(output, skip_special_tokens=True)
                    )
                ],
                -1,
            )
            for output in outputs
        ]


class CodeShellMaskPredictModel(MaskPredictModel):
    NAME = "codeshell"
    PARAMETER_DATATYPE = ParameterDataType.BF16

    VARIANTS = [
        MaskPredictModelVariant("7B", default_top_p=0.6, default_temperature=0.7)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codeshell: CodeShell = CodeShell(self.load_config, f"wisdomshell/codeshell-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codeshell.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return "<fim_suffix>"
