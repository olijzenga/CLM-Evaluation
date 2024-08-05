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
from transformers import AutoTokenizer, T5ForConditionalGeneration, RobertaTokenizer

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant
from clm.clms.stopping_conditions import StoppingCriteriaContainsToken


class CodeT5:
    MASK_TOKEN_PATTERN = r"<extra_id_\d+>"

    def __init__(self, load_config: ModelLoadConfig, model_name: str = "Salesforce/codet5-large") -> None:
        self.device: torch.device = load_config.device
        self.tokenizer: RobertaTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
            model_name, **load_config.get_load_parameters()
        )

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        source_ids = input_ids.clone().to(self.device)

        # simply generate a single sequence
        generate_ids = self.model.generate(
            source_ids,
            **sampling_config.get_sampling_parameters(),
            # Needs force stop at next mask, otherwise it keeps going
            stopping_criteria=StoppingCriteriaContainsToken(
                [self.tokenizer.convert_tokens_to_ids("<extra_id_1>")]
            ).to_list(),
        )
        results = [self.tokenizer.decode(generated_id) for generated_id in generate_ids]

        return [MaskPredictResult(None, self._split(result, text), -1.0) for result in results]

    def _split(self, output: str, text: str) -> list[str]:
        assert isinstance(output, str)
        assert isinstance(text, str)

        nr_prompt_masks = len(re.findall(self.MASK_TOKEN_PATTERN, text))

        mask_matches = re.findall(self.MASK_TOKEN_PATTERN, output)
        separator = "|||||||SEPARATOR||||||||"  # No way this actually appears in real code right? :)

        for match in mask_matches:
            output = output.replace(match, separator)

        result = []
        # Ignore first one since its empty or contains junk, ignore subsequent ones since they were not requested
        for match in output.split(separator)[1 : nr_prompt_masks + 1]:
            for token in ("<s>", "</s>", "<pad>"):
                match = match.replace(token, "")
            result.append(match)

        return result


class CodeT5MaskPredictModel(MaskPredictModel):
    NAME = "codet5"
    PARAMETER_DATATYPE = ParameterDataType.F32

    VARIANTS = [
        MaskPredictModelVariant("large", default_top_p=0.2, default_temperature=0.1),
        MaskPredictModelVariant("base", default_beam_size=5),
        MaskPredictModelVariant("small", default_beam_size=10)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codet5: CodeT5 = CodeT5(self.load_config, f"Salesforce/codet5-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codet5.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return f"<extra_id_{self.MASK_NR_TEMPLATE}>"
