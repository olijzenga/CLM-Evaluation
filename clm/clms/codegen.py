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


class CodeGen:
    def __init__(
        self, load_config: ModelLoadConfig, model_name: str = "salesforce/codegen-350m-multi"
    ) -> None:
        self.device = load_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_config.get_load_parameters())

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        print(self.tokenizer.eos_token)
        # eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        eos_id = self.tokenizer.convert_tokens_to_ids("}")
        pad_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            **sampling_config.get_sampling_parameters(),
        )

        result = []
        for generated_id in generated_ids:
            result.append(
                MaskPredictResult(self.tokenizer.decode(generated_id, skip_special_tokens=True), None, -1)
            )

        return result


class CodeGenMaskPredictModel(MaskPredictModel):
    NAME = "codegen"
    PARAMETER_DATATYPE = ParameterDataType.F16

    VARIANTS = [
        MaskPredictModelVariant("350M-multi", default_beam_size=10),
        MaskPredictModelVariant("2B-multi", default_beam_size=10),
        MaskPredictModelVariant("6B-multi", default_beam_size=10),
        MaskPredictModelVariant("16B-multi", default_beam_size=10),
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codegen = CodeGen(self.load_config, f"salesforce/codegen-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codegen.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return ""

    @staticmethod
    def get_model_variants() -> list[str]:
        return ["350M-multi", "2B-multi", "6B-multi", "16B-multi"]
