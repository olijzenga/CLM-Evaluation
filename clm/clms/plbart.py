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
from transformers import PLBartTokenizer, PLBartForConditionalGeneration

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm import util
from clm.model.mask_predict import MaskPredictModelVariant


class PLBART:
    def __init__(self, load_config: ModelLoadConfig, model_name: str = "uclanlp/plbart-base") -> None:
        self.tokenizer = PLBartTokenizer.from_pretrained(model_name, src_lang="java", tgt_lang="java")
        self.model = PLBartForConditionalGeneration.from_pretrained(model_name, **load_config.get_load_parameters())

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer(f"<s>{text} </s>java", add_special_tokens=False, return_tensors="pt").input_ids
        source_ids = input_ids.clone().to(self.model.device)

        # simply generate a single sequence. For some reason max_new_tokens does not work for PLBART so we do it manually
        sampling_parameters = sampling_config.get_sampling_parameters()
        sampling_parameters['max_length'] = len(input_ids[0]) + sampling_config.max_new_tokens
        sampling_parameters['max_new_tokens'] = None

        generated_ids = self.model.generate(
            source_ids,
            decoder_start_token_id=self.tokenizer.lang_code_to_id["__java__"],
            **sampling_parameters,
        )

        return [
            MaskPredictResult(
                # PLBART returns unformatted java code so we fix it using a simple indenter function
                util.format_java(self.tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)),
                None,
                -1,
            )
            for generated_id in generated_ids
        ]


class PLBARTMaskPredictModel(MaskPredictModel):
    NAME = "plbart"
    PARAMETER_DATATYPE = ParameterDataType.F32

    VARIANTS = [
        MaskPredictModelVariant("base", default_beam_size=10),
        MaskPredictModelVariant("large", default_beam_size=10)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.plbart = PLBART(self.load_config, f"uclanlp/plbart-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.plbart.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return "<mask>"

    @staticmethod
    def get_model_variants() -> list[str]:
        return ["base", "large"]
