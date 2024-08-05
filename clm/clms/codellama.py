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


class CodeLlama:
    def __init__(self, load_config: ModelLoadConfig, model_name: str = "codellama/codellama-7b-hf") -> None:
        self.device = load_config.device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **load_config.get_load_parameters()
        )

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.cuda()
        generated_ids = self.model.generate(
            input_ids,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            **sampling_config.get_sampling_parameters(),
        )

        return [
            MaskPredictResult(
                None,
                [self.tokenizer.decode(generated_id[len(input_ids[0]) :], skip_special_tokens=True)],
                -1.0,
            )
            for generated_id in generated_ids
        ]


class CodeLlamaMaskPredictModel(MaskPredictModel):
    NAME = "codellama"
    PARAMETER_DATATYPE = ParameterDataType.BF16

    VARIANTS = [
        MaskPredictModelVariant("7B", default_top_p=0.6, default_temperature=1.0),
        MaskPredictModelVariant("13B", default_top_p=0.8, default_temperature=0.7),
        MaskPredictModelVariant("7B-instruct", default_top_p=0.8, default_temperature=1.0),
        MaskPredictModelVariant("13B-instruct", default_top_p=0.8, default_temperature=1.3)
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codellama = CodeLlama(self.load_config, f"codellama/codellama-{self.model_variant.name}-hf")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codellama.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return f"<FILL_ME>"
