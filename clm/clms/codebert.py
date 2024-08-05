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

from transformers import RobertaForMaskedLM, RobertaTokenizer, pipeline

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant


class CodeBERT:
    def __init__(self, load_config: ModelLoadConfig, model_variant="base") -> None:
        model_name = f"microsoft/codebert-{model_variant}-mlm"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaForMaskedLM.from_pretrained(model_name, **load_config.get_load_parameters())
        self.device = load_config.device

        self._fill_mask_pipeline = pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        self._fill_mask_pipeline.device = load_config.device

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        nr_mask_predictions = 0
        stop = False
        results: list[tuple[MaskPredictResult, list[float]]] = [(MaskPredictResult(text, [], -1.0), [])]
        while not stop:
            results_: list[tuple[MaskPredictResult, list[float]]] = []

            for existing_result, scores in results:
                new_results = list(self._fill_mask_pipeline(existing_result.full_result))
                nr_mask_predictions += 1

                if isinstance(new_results[0], dict):
                    stop = True
                else:
                    new_results = new_results[0]

                for new_result in new_results:
                    text_ = self._strip(new_result["sequence"])
                    mask_replacement = self._find_mask_replacement(existing_result.full_result, text_)

                    results_.append(
                        (
                            MaskPredictResult(
                                text_, existing_result.mask_replacements + [mask_replacement], -1
                            ),
                            scores + [new_result["score"]],
                        )
                    )

            # TODO: implement logarithmic scoring similar to AlphaRepair (I think)
            results_.sort(key=lambda r: sum(r[1]) / len(r[1]), reverse=True)
            results = results_[: sampling_config.num_return_sequences]

        print(f"CodeBERT did a total of {nr_mask_predictions} mask predictions")

        return [
            MaskPredictResult(None, result.mask_replacements, sum(scores) / len(scores))
            for result, scores in results
        ]

    def _find_mask_replacement(self, old_text: str, new_text: str) -> str:
        """Returns a single mask replacement based on the input and output of a single mask prediction"""
        assert isinstance(old_text, str)
        assert isinstance(new_text, str)

        prefix, suffix = old_text.split("<mask>", 1)

        replacement = new_text[len(prefix) :]

        # CodeBERT removes whitespaces between masks to match that here
        while " <mask>" in suffix:
            suffix = suffix.replace(" <mask>", "<mask>")

        # print('='*20)
        # print("old", old_text, "end", len(old_text))
        # print("new", new_text, "new", len(new_text))
        # print("suffix", suffix.replace(' \t\n', 'WS'), "end", len(suffix))
        # print("preresult", replacement, 'end', len(replacement))

        if len(suffix) != 0:
            replacement = replacement[: -len(suffix)]

        return replacement

    def _strip(self, text: str) -> str:
        if text.startswith("<s>") and text.endswith("</s>"):
            return text[3:-4]
        return text


class CodeBERTMaskPredictModel(MaskPredictModel):
    NAME = "codebert"
    PARAMETER_DATATYPE = ParameterDataType.F32

    VARIANTS = [
        MaskPredictModelVariant("base", default_beam_size=10)
    ]

    def __init__(
        self,
        load_config: ModelLoadConfig,
        model_variant: MaskPredictModelVariant | None,
        min_tokens: int = 1,
        max_tokens: int = 1,
    ) -> None:
        super().__init__(load_config, model_variant)

        self.min_tokens: int = min_tokens
        self.max_tokens: int = max_tokens
        self.codebert: CodeBERT = CodeBERT(self.load_config)

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        results = []
        for nr_masks in range(self.min_tokens, self.max_tokens + 1):
            print(f"Generating with {nr_masks} masks")

            text_ = text.replace("<mask>", "<mask>" * nr_masks)
            results.extend(self.codebert.fill_mask(text_, sampling_config))

        return results

    def get_does_multi_token_prediction(self) -> bool:
        return False

    def get_mask(self, mask: str) -> str:
        return "<mask>"
