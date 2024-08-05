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

from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    RobertaTokenizerFast,
    AutoModelForSeq2SeqLM,
)

from clm.model import (
    MaskPredictResult,
    MaskPredictModel,
    ModelSamplingConfig,
    ModelLoadConfig,
    ParameterDataType,
)
from clm.model.mask_predict import MaskPredictModelVariant


class CodeT5p:
    MASK_TOKEN_PATTERN = r"<extra_id_\d+>"

    def __init__(self, load_config: ModelLoadConfig, model_name: str = "Salesforce/codet5p-220m") -> None:
        self.tokenizer: RobertaTokenizerFast = AutoTokenizer.from_pretrained(model_name)
        if "220m" in model_name or "770m" in model_name:
            self.model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, trust_remote_code=True, **load_config.get_load_parameters()
            )
        self.device: torch.device = load_config.device

    def fill_mask(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids

        source_ids = input_ids.clone().to(self.device)

        # simply generate a single sequence
        generate_ids = self.model.generate(
            source_ids, **sampling_config.get_sampling_parameters()
        )
        results = [self.tokenizer.decode(generated_id) for generated_id in generate_ids]

        return self._post_process_results(text, results)

    def _post_process_results(
        self, original_text: str, decoder_results: list[str]
    ) -> list[MaskPredictResult]:
        processed_results = []
        for result in decoder_results:
            processed_result = self._split(original_text, result)
            if processed_result is None:
                continue

            processed_results.append(MaskPredictResult(None, processed_result, -1.0))
        return processed_results

    def _split(self, original_text: str, result_text: str) -> list[str] | None:
        """Returns null if the model does not return a replacement for each mask"""
        assert isinstance(result_text, str)

        mask_replacements = self._get_mask_replacements(result_text)
        # For some reason CodeT5p sometimes includes a new mask at the end of its output. Remove masks which
        # were not in the input here to avoid returning an unexpected number of token replacments
        mask_replacements = {m: r for m, r in mask_replacements.items() if m in original_text}

        # Check if replacements were generated for all masks, if not return None as the result is unusable
        original_text_masks = self._get_mask_tokens(original_text)
        if set(original_text_masks) != set(mask_replacements.keys()):
            return None

        # Ignore first one since its empty or contains junk
        result = []
        for mask, mask_replacement in mask_replacements.items():
            for token in ("<s>", "</s>", "<pad>"):
                # Remove special tokens from output
                mask_replacement = mask_replacement.replace(token, "")

            if mask_replacement.startswith("\n"):
                mask_replacement = mask_replacement[1:]

            # Remove start of mask line from result since we should return only what should be in place of
            # the mask, not what should be before it.
            prefix = self._get_mask_prefix(mask, original_text)
            if mask_replacement.startswith(prefix):
                mask_replacement = mask_replacement[len(prefix) :]

            result.append(mask_replacement)

        return result

    def _get_mask_replacements(self, text: str) -> dict:
        """
        Example: for input below returns {"<extra_id_0>": "     n >>= 1;", "<extra_id_1>": "blablablabla"}
        <pad><extra_id_0>        n >>= 1;<extra_id_1>blablabla
        """
        masks = self._get_mask_tokens(text)
        mask_matches: list[re.Match] = [re.search(mask, text) for mask in masks]  # type: ignore
        assert all(match is not None for match in mask_matches)
        mask_matches.sort(key=lambda m: m.span()[0])

        result = {}
        remaining_text = text
        for i, mask_match in enumerate(mask_matches):
            mask_text = mask_match.group(0)
            _, remaining_text = remaining_text.split(mask_text, 1)

            if i + 1 == len(mask_matches):
                result[mask_text] = remaining_text
                break

            # Split text at starting point of next mask token
            next_match = mask_matches[i + 1]
            current_text = remaining_text.split(next_match.group(0))[0]
            remaining_text = remaining_text[len(current_text) :]

            result[mask_text] = current_text

        return result

    def _get_mask_prefix(self, mask: str, text: str) -> str:
        for line in text.split("\n"):
            if mask in line:
                return line.split(mask, 2)[0]
        raise Exception()

    def _get_mask_tokens(self, text: str) -> list[str]:
        return re.findall(self.MASK_TOKEN_PATTERN, text)


class CodeT5pMaskPredictModel(MaskPredictModel):
    NAME = "codet5p"
    PARAMETER_DATATYPE = ParameterDataType.F16

    VARIANTS = [
        MaskPredictModelVariant("220m", default_beam_size=10),
        MaskPredictModelVariant("220m-py", default_beam_size=10),
        MaskPredictModelVariant("770m", default_beam_size=10),
        MaskPredictModelVariant("2B", default_beam_size=10),
        MaskPredictModelVariant("6B", default_beam_size=10),
        MaskPredictModelVariant("16B", default_beam_size=10),
    ]

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        super().__init__(load_config, model_variant)

        self.codet5p = CodeT5p(self.load_config, f"Salesforce/codet5p-{self.model_variant.name}")

    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        return self.codet5p.fill_mask(text, sampling_config)

    def get_does_multi_token_prediction(self) -> bool:
        return True

    def get_mask(self, mask: str) -> str:
        return f"<extra_id_{self.MASK_NR_TEMPLATE}>"
