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

from transformers import AutoTokenizer, AutoModelForCausalLM

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant
from clm.clms.stopping_conditions import StoppingCriteriaContainsString


class CodeGen2:
    def __init__(self, load_config: ModelLoadConfig, model_name: str = "salesforce/codegen2-1B") -> None:
        self.device = load_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, revision="main", **load_config.get_load_parameters()
        )

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaContainsString(['<eom>'], self.tokenizer).to_list(),
            **sampling_config.get_sampling_parameters(),
        )

        result = []
        for generated_id in generated_ids:
            mask_replacement = self._parse_result(self.tokenizer.decode(generated_id, skip_special_tokens=True))
            if mask_replacement is None:
                continue

            result.append(MaskPredictResult(None, [mask_replacement], -1))

        return result

    @staticmethod
    def _parse_result(result: str) -> str | None:
        if '<sep>' not in result:
            return None

        result = result.split('<sep>', 1)[1]
        result = result.replace('<mask_1>', '').replace('<eom>', '')

        return result


class CodeGen2MaskPredictModel(MaskPredictModel):
    NAME = "codegen2"
    PARAMETER_DATATYPE = ParameterDataType.F32

    VARIANTS = [
        MaskPredictModelVariant("1B", default_top_p=0.6, default_temperature=0.7),
        MaskPredictModelVariant("3_7B", default_top_p=0.4, default_temperature=1.3),
        MaskPredictModelVariant("7B", default_top_p=0.8, default_temperature=0.1),
        MaskPredictModelVariant("16B", default_top_p=0.4, default_temperature=1.9)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codegen2 = CodeGen2(self.load_config, f"salesforce/codegen2-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codegen2.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return f"<mask_{self.MASK_NR_TEMPLATE}>"

    def get_input_suffix(self, text: str) -> str:
        if "<mask_1>" in text:
            return "<|endoftext|><sep><mask_1>"

        return ""

    def get_first_token_number(self) -> int:
        return 1
