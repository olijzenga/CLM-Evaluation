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

import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from clm.model import (
    MaskPredictModel,
    MaskPredictResult,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant
from clm.clms.stopping_conditions import StoppingCriteriaContainsString
from clm import util


class InCoder:
    """Example usage: https://github.com/dpfried/incoder/blob/main/example_usage.py"""

    EOT = "<|endoftext|>"
    EOM = "<|endofmask|>"
    MASK0 = "<|mask:0|>"

    def __init__(self, load_config: ModelLoadConfig, model_name: str = "facebook/incoder-1B") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_config.get_load_parameters())
        self.device: torch.device = load_config.device

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        if "<|mask:1|>" in text:
            raise Exception("InCoder cannot be used with multiple masks")

        suffix = text.split(self.MASK0, 1)[1]

        # This is needed to make InCoder generate the EOM token after the mask replacement. This allows
        # us to figure out which code to replace.
        text += "<|mask:1|><|mask:0|>"

        eos_id = self.tokenizer.convert_tokens_to_ids('</code>')

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids.cuda()

        generated_ids = self.model.generate(
            input_ids=input_ids,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            output_scores=True,
            return_dict_in_generate=True,
            stopping_criteria=[
                StoppingCriteriaContainsString([suffix], self.tokenizer, self.MASK0, stop_after=True),
                StoppingCriteriaContainsString(["<|endofmask|>", suffix], self.tokenizer, self.MASK0, stop_after=False)
            ],
            **sampling_config.get_sampling_parameters(),
        )

        results = []
        for sequence, score in zip(generated_ids.sequences, generated_ids.scores):
            mask_replacement = self.tokenizer.decode(sequence[len(input_ids[0]):], clean_up_tokenization_spaces=False, skip_special_tokens=False)
            mask_replacement = mask_replacement.replace(self.EOM, "").replace(self.EOT, "").replace("</code>", "")
            
            results.append(
                MaskPredictResult(
                    None,
                    [mask_replacement],
                    -1.0
                )
            )

        return results


class InCoderMaskPredictModel(MaskPredictModel):
    NAME = "incoder"
    PARAMETER_DATATYPE = ParameterDataType.F16

    VARIANTS = [
        MaskPredictModelVariant("6B", default_beam_size=10),
        MaskPredictModelVariant("1B", default_beam_size=10)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.incoder = InCoder(self.load_config, f"facebook/incoder-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.incoder.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return f"<|mask:{self.MASK_NR_TEMPLATE}|>"
