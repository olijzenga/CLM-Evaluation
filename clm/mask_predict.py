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
import time
import fire
import logging
import torch

from clm import util
from clm.model import (
    MaskPredictResult,
    NormalizedMaskPredictResult,
    MaskPredictModel,
    MaskReplacement,
    ModelSamplingConfig,
    QuantizationMode,
    ModelSamplingPreferences,
    ModelLoadPreferences
)
from clm.clms import MaskPredictModelFactory
from clm.setup import setup_logging

log = logging.getLogger(__name__)


class MaskPredictException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class MaskPredictor:
    def __init__(self, text: str, model: MaskPredictModel, sampling_config: ModelSamplingConfig) -> None:
        assert isinstance(text, str)
        assert isinstance(model, MaskPredictModel)
        assert isinstance(sampling_config, ModelSamplingConfig)

        self._raw_text: str = text
        self._model: MaskPredictModel = model
        self._sampling_config: ModelSamplingConfig = sampling_config

        # Input text provided to the CLM
        self._model_specific_tokens_text: str = ""
        # Mapping of generic mask tokens to model specific mask tokens
        self._model_specific_tokens_map: list[tuple[str, str]] = []

        self._results: list[MaskPredictResult] = []
        self._normalized_results: list[NormalizedMaskPredictResult] = []

    def predict(self) -> list[NormalizedMaskPredictResult]:
        self._create_model_specific_tokens_text()

        log.debug("Actual input text:\n" + self._model_specific_tokens_text)

        if len(self._model_specific_tokens_map) == 0:
            log.warning("found no valid mask tokens to replace")

        log.info(f"Sampling {self._model} with parameters {self._sampling_config.get_sampling_parameters()}")
        self._results = self._model.predict(self._model_specific_tokens_text, self._sampling_config)
        self._normalized_results = self._normalize_results(self._results)

        return self._normalized_results

    def _get_mask_pattern(self) -> str:
        if self._model.get_does_multi_token_prediction():
            # Match sequences of one or more mask tokens at once to collapse them into one
            return MaskPredictModel.GENERIC_MASK_SEQUENCE_PATTERN
        else:
            # Match individual mask tokens
            return MaskPredictModel.ALL_GENERIC_MASKS_PATTERN

    def _match_mask_replacements(self, full_result: str) -> list[str]:
        """Best effort replacement to mask matcher. Probably doesn't work for complex cases."""
        if len(self._model_specific_tokens_map) == 1:
            # Use simple prefix and suffix matcher in case of a single mask token
            generic_mask, native_mask = self._model_specific_tokens_map[0]
            prefix, suffix = self._raw_text.split(generic_mask, 1)

            def _reverse_str(s):
                return "".join(reversed(s))

            without_prefix = util.remove_prefix_ignoring_whitespaces(prefix, full_result)
            replacement = _reverse_str(util.remove_prefix_ignoring_whitespaces(_reverse_str(suffix), _reverse_str(without_prefix)))
            return [replacement.strip()]

        # Replace each mask with a wildcard matching group
        original_text_pattern = self._raw_text
        for mask, _ in self._model_specific_tokens_map:
            original_text_pattern = original_text_pattern.replace(mask, "(.*)", 1)

        # Now match wildcards to result text using regex search
        match = re.search(original_text_pattern, full_result)
        if match is None:
            raise MaskPredictException(
                f"Could not match full result {full_result} to masks {self._model_specific_tokens_map} "
                f"with pattern {original_text_pattern}"
            )

        mask_replacements = []
        for i in range(len(self._model_specific_tokens_map)):
            mask_replacements.append(match.group(i + 1))
        return mask_replacements

    def _normalize_results(self, results: list[MaskPredictResult]) -> list[NormalizedMaskPredictResult]:
        normalized_results = []

        for result in results:
            if result.full_result is not None:
                # Extract mask replacements from full result and then drop the full result as
                # we cannot trust CLMs to preserve code formatting
                result = MaskPredictResult(
                    None,
                    [r for r in self._match_mask_replacements(result.full_result) ],
                    result.confidence
                )

            if result.mask_replacements is None:
                raise MaskPredictException(
                    f"Expected mask predict result {result} to have either a full result or "
                    "mask replacements"
                )

            if len(result.mask_replacements) != len(self._model_specific_tokens_map):
                raise MaskPredictException(
                    f"Mask predict result {result} contains {len(result.mask_replacements)} mask "
                    f"replacements but expected {len(self._model_specific_tokens_map)}"
                )

            replacements = []
            for (mask, native_mask), replacement_str in zip(
                self._model_specific_tokens_map, result.mask_replacements
            ):
                replacements.append(MaskReplacement(mask, native_mask, replacement_str))

            full_result = self._raw_text
            for replacement in replacements:
                full_result = full_result.replace(replacement.mask, replacement.replacement, 1)

            normalized_results.append(
                NormalizedMaskPredictResult(full_result, replacements, result.confidence)
            )

        return normalized_results

    def _create_model_specific_tokens_text(self) -> None:
        """
        Create input text with mask tokens specific to the mask prediction model being used
        """
        text = self._raw_text
        numbered_mask_ctr = self._model.get_first_token_number()
        self._model_specific_tokens_text = ""
        self._model_specific_tokens_map = []
        while True:
            match = re.search(self._get_mask_pattern(), text)
            if match is None:
                self._model_specific_tokens_text += text
                break

            generic_mask = match.group(0)
            specific_mask = self._model.get_mask(generic_mask)

            if MaskPredictModel.MASK_NR_TEMPLATE in specific_mask:
                specific_mask = specific_mask.replace(
                    MaskPredictModel.MASK_NR_TEMPLATE, str(numbered_mask_ctr)
                )
                numbered_mask_ctr += 1

            self._model_specific_tokens_map.append((generic_mask, specific_mask))
            self._model_specific_tokens_text += text[: match.span()[0]] + specific_mask
            text = text[match.span()[1] :]

        self._model_specific_tokens_text += self._model.get_input_suffix(self._model_specific_tokens_text)


def _print_results(results: list[NormalizedMaskPredictResult]) -> None:
    print("Results:")

    for i, result in enumerate(results):
        if result.full_result.strip() == "":
            continue

        header_text = f" Result {i + 1} confidence {round(result.confidence, 3)} "
        line_length = max(util.get_max_line_length(result.full_result) + 2, len(header_text) + 4)
        print(util.center_text(header_text, "=", line_length))
        print(result.full_result)

        separator = "=" * line_length
        print(separator)

        for mask_replacement in result.mask_replacements:
            print(f'"{mask_replacement.mask}"', "=>", f'"{mask_replacement.replacement}"')

        print(separator)


def _predict_mask(text: str, model: MaskPredictModel, sampling_preferences: ModelSamplingPreferences) -> None:
    print(f"Mask prediction using {model.__class__.__name__} {model.model_variant.name}")

    predictor = MaskPredictor(
        text, model, ModelSamplingConfig.create_for_preferences_and_model(sampling_preferences, model)
    )

    start_time = time.time()
    results = predictor.predict()
    mask_predict_time = time.time() - start_time

    results.sort(key=lambda r: r.confidence, reverse=True)
    _print_results(results)
    print(f"Mask prediction took {round(mask_predict_time, 1)} seconds")


def _get_text(file_path: str | None, raw_text: str | None) -> str:
    if file_path is None and raw_text is None:
        print("Either a file path or raw text must be provided")
        exit(1)

    if raw_text is not None:
        return raw_text

    with open(file_path, "r") as f:  # type: ignore
        return f.read()


def main(
    model_name: str,
    file_path: str | None = None,
    raw_text: str | None = None,
    device: str | None = None,
    model_variant: str | None = None,
    nr_results: int | None = None,
    num_beams: int | None = None,
    top_p: float | None = None,
    temperature: float | None = None,
    quantization_mode: str | None = None,
    seed: int = 0,
) -> None:
    """
    Perform a single mask predict task and print the result

    Args:
        model_name: Name of the CLM.
        file_path: Path to the file containing the prompt.
        raw_text: The prompt. Can optionally be used instead of file_path.
        device: Device on which the CLM is loaded. Either 'cuda' for GPU or 'cpu' for CPU.
        model_variant: Variant of the CLM (i.e. small, base, large).
        nr_results: Number of infills to generate.
        num_beams: Beam size for beam search.
        top_p: Top p for nucleus sampling.
        temperature: Temperature for nucleus sampling.
        quantization_mode: Quantization mode, one of 4bit, 8bit, 16bit, 32bit. Leave empty to omit quantization.
        seed: Seed used for sampling the CLM.
    """

    setup_logging()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if quantization_mode is not None:
        quantization_mode = QuantizationMode.from_string(quantization_mode)

    load_preferences = ModelLoadPreferences(quantization_mode, device)
    sampling_preferences = ModelSamplingPreferences(
        num_return_sequences=nr_results,
        num_beams=num_beams,
        top_p=top_p,
        temperature=temperature,
    )

    text = _get_text(file_path, raw_text)
    model = MaskPredictModelFactory(load_preferences).get_model(model_name, model_variant)

    _predict_mask(text, model, sampling_preferences)


if __name__ == "__main__":
    fire.Fire(main)
