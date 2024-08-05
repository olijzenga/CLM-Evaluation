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
import logging
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from transformers import BitsAndBytesConfig

log = logging.getLogger(__name__)


@dataclass
class MaskPredictResult:
    full_result: str | None
    mask_replacements: list[str] | None
    confidence: float


@dataclass
class MaskReplacement:
    mask: str
    native_mask: str
    replacement: str


@dataclass
class NormalizedMaskPredictResult:
    full_result: str
    mask_replacements: list[MaskReplacement]
    confidence: float

    def __init__(self, full_result: str, mask_replacements: list[MaskReplacement], confidence: float):
        self.full_result = full_result
        self.mask_replacements = mask_replacements
        self.confidence = round(confidence, 3)

    def get_full_result(self, original_text: str) -> str:
        for replacement in self.mask_replacements:
            original_text = original_text.replace(replacement.mask, replacement.replacement)
        return original_text


class ModelSamplingPreferences:
    def __init__(
        self,
        num_return_sequences: int | None = None,
        do_beam_search: bool | None = None,
        num_beams: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
    ):
        assert num_return_sequences is None or isinstance(num_return_sequences, int)
        assert do_beam_search is None or isinstance(do_beam_search, bool)
        assert num_beams is None or isinstance(num_beams, int)
        assert top_p is None or isinstance(top_p, float)
        assert temperature is None or isinstance(temperature, float)

        if do_beam_search is not None:
            if do_beam_search:
                assert (
                    top_p is None and temperature is None
                ), "Cannot combine nucleus sampling with beam search"
            else:
                assert num_beams is None, "Cannot combine nucleus sampling with beam search"

        self.num_return_sequences: int | None = num_return_sequences
        self.do_beam_search: bool | None = do_beam_search
        self.num_beams: int | None = num_beams
        self.top_p: float | None = top_p
        self.temperature: float | None = temperature


class ModelSamplingConfig:
    def __init__(
        self, num_return_sequences: int, num_beams: int | None, top_p: float | None, temperature: float | None
    ):
        assert isinstance(num_return_sequences, int)
        assert num_beams is None or isinstance(num_beams, int)
        assert top_p is None or isinstance(top_p, float)
        assert temperature is None or isinstance(temperature, float)

        if num_beams is None:
            assert (
                top_p is not None and temperature is not None
            ), "both top_p and temperature must be defined when using nucleus sampling"
        else:
            assert (
                num_return_sequences <= num_beams
            ), "num_beams must be at greater than or equal to num_return_sequences"
            assert top_p is None and temperature is None, "cannot combine nucleus sampling and beam search"

        if top_p is not None:
            assert 0 <= top_p <= 1, "top_p must be between 0 and 1"

        if temperature is not None:
            assert temperature > 0, "temperature must be positive"

        self.num_return_sequences: int = num_return_sequences
        self.num_beams: int | None = num_beams
        self.top_p: float | None = top_p
        self.temperature: float | None = temperature
        self.max_new_tokens: int = 128

    def get_sampling_parameters(self) -> dict:
        # Providing values of each parameter instead of using None is required to avoid warnings.
        do_sample = self.top_p is not None or self.temperature is not None
        result = {
            "max_new_tokens": 128,
            "num_return_sequences": self.num_return_sequences,
            "do_sample": do_sample
        }

        if do_sample:
            result['top_p'] = self.top_p or 1.0
            result['temperature'] = self.temperature or 1.0
        else:
            result['num_beams'] = self.num_beams or 1
            
        return result

    @staticmethod
    def create_for_preferences_and_model(
        preferences: ModelSamplingPreferences, model: "MaskPredictModel"
    ) -> "ModelSamplingConfig":
        assert isinstance(preferences, ModelSamplingPreferences)
        assert isinstance(model, MaskPredictModel)

        variant: MaskPredictModelVariant = model.model_variant

        num_return_sequences = preferences.num_return_sequences or 10

        if preferences.do_beam_search is None:
            if preferences.num_beams is not None:
                do_beam_search = True
            elif preferences.top_p is not None or preferences.temperature is not None:
                do_beam_search = True
            else:
                do_beam_search = variant.default_beam_size is not None
        else:
            do_beam_search = preferences.do_beam_search

        num_beams = None
        top_p = None
        temperature = None

        if do_beam_search:
            if preferences.num_beams is not None:
                num_beams = preferences.num_beams
            elif variant.default_beam_size is not None:
                num_beams = max(num_return_sequences, variant.default_beam_size)
            else:
                num_beams = num_return_sequences
        else:
            top_p = preferences.top_p or variant.default_top_p
            temperature = preferences.temperature or variant.default_temperature

        return ModelSamplingConfig(num_return_sequences, num_beams, top_p, temperature)


class QuantizationMode(Enum):
    B32 = "32bit"
    B16 = "16bit"
    B8 = "8bit"
    B4 = "4bit"

    @staticmethod
    def from_string(value: str) -> "QuantizationMode":
        for option in QuantizationMode:
            if value == option.value:
                return option
        raise ValueError(f"Must be one of {[e.value for e in QuantizationMode]}, got {value}")

    def nr_bits(self) -> int:
        match self:
            case QuantizationMode.B32:
                return 32
            case QuantizationMode.B16:
                return 16
            case QuantizationMode.B8:
                return 8
            case QuantizationMode.B4:
                return 4
        raise NotImplementedError(self)


class ParameterDataType(Enum):
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"

    def nr_bits(self) -> int:
        match self:
            case ParameterDataType.F32:
                return 32
            case ParameterDataType.F16 | ParameterDataType.BF16:
                return 16
            case ParameterDataType.INT8:
                return 8
            case ParameterDataType.INT4:
                return 4
        raise NotImplementedError(self)

    def get_torch_dtype(self) -> torch.dtype | None:
        match self:
            case ParameterDataType.F32:
                return torch.float32
            case ParameterDataType.BF16:
                return torch.bfloat16
            case ParameterDataType.F16:
                return torch.float16
        return None


@dataclass
class ModelLoadPreferences:
    def __init__(self, quantization_mode: QuantizationMode | None = None, device_name: str | None = None):
        assert quantization_mode is None or isinstance(quantization_mode, QuantizationMode)
        assert device_name is None or isinstance(device_name, str)
        assert device_name is None or device_name in ("cpu", "cuda")

        if quantization_mode is None:
            self.quantization_mode = QuantizationMode.B16
        else:
            self.quantization_mode = quantization_mode

        if device_name is None:
            self.device_name = "cuda"
        else:
            self.device_name = device_name

    def to_json(self) -> dict:
        return {"quantization_mode": self.quantization_mode.value, "device_name": self.device_name}


@dataclass
class ModelLoadConfig:
    torch_dtype: torch.dtype | None
    quantization_config: BitsAndBytesConfig | None
    device: torch.device | None
    device_map: str | None

    def get_load_parameters(self) -> dict:
        return {
            "torch_dtype": self.torch_dtype,
            "quantization_config": self.quantization_config,
            "device_map": self.device_map,
        }

    @staticmethod
    def create_for_preferences_and_model(
        preferences: ModelLoadPreferences, model_cls: type["MaskPredictModel"]
    ) -> "ModelLoadConfig":
        assert isinstance(preferences, ModelLoadPreferences)
        assert isinstance(model_cls, type)
        assert issubclass(model_cls, MaskPredictModel)

        torch_dtype = model_cls.PARAMETER_DATATYPE.get_torch_dtype()
        bitsandbytes_config = None

        # Only apply quantization if its going to reduce the size of the model, and when using GPU
        if (
            preferences.device_name != "cpu"
            and preferences.quantization_mode.nr_bits() < model_cls.PARAMETER_DATATYPE.nr_bits()
        ):
            match preferences.quantization_mode:
                case QuantizationMode.B32:
                    torch_dtype = torch.float32
                case QuantizationMode.B16:
                    torch_dtype = torch.float16
                case QuantizationMode.B8:
                    log.info("Using 8bit quantization")
                    bitsandbytes_config = BitsAndBytesConfig(load_in_8bit=True)
                case QuantizationMode.B4:
                    log.info("Using 4bit quantization")
                    bitsandbytes_config = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                    )
                case _:
                    raise NotImplementedError(preferences.quantization_mode)

        if preferences.device_name == "cuda":
            device_map = "auto"
        else:
            device_map = preferences.device_name

        return ModelLoadConfig(
            torch_dtype, bitsandbytes_config, torch.device(preferences.device_name), device_map
        )

    def to_json(self) -> dict:
        quantization_config = None
        if self.quantization_config is not None:
            quantization_config = {}
            for k, v in self.quantization_config.__dict__.items():
                if isinstance(v, torch.dtype):
                    quantization_config[k] = str(v)
                else:
                    quantization_config[k] = v

        return {
            "torch_dtype": str(self.torch_dtype),
            "quantization_config": quantization_config,
            "device": str(self.device),
            "device_map": self.device_map,
        }


@dataclass
class MaskPredictModelVariant:
    name: str
    default_top_p: float | None = None
    default_temperature: float | None = None
    default_beam_size: int | None = None


class MaskPredictModel(ABC):
    GENERIC_MASKS = [GENERIC_MASK_ANY := "<mask>", GENERIC_MASK_ID := "<mask_id>"]

    ALL_GENERIC_MASKS_PATTERN = "|".join(re.escape(generic_mask) for generic_mask in GENERIC_MASKS)
    GENERIC_MASK_SEQUENCE_PATTERN = rf"({ALL_GENERIC_MASKS_PATTERN})(\s*({ALL_GENERIC_MASKS_PATTERN}))*"

    MASK_NR_TEMPLATE = "|NR|"

    NAME: str = None
    PARAMETER_DATATYPE: ParameterDataType = None

    VARIANTS: list[MaskPredictModelVariant] = []

    def __init__(self, load_config: ModelLoadConfig, model_variant: MaskPredictModelVariant | None = None) -> None:
        assert self.NAME is not None, "NAME must be set for MaskPredictModel implementations"
        assert (
            self.PARAMETER_DATATYPE is not None
        ), "PARAMETER_DATATYPE must be set for MaskPredictModel implementations"

        super().__init__()
        assert isinstance(load_config, ModelLoadConfig)
        assert model_variant is None or isinstance(model_variant, MaskPredictModelVariant)

        if model_variant is None:
            model_variant = self.get_default_variant()

        self.device: torch.device = load_config.device
        self.load_config: ModelLoadConfig = load_config

        self.model_variant: MaskPredictModelVariant = model_variant
        self.model_name: str = self.__class__.__name__[: -len("MaskPredictModel")].lower()

    @abstractmethod
    def predict(self, text: str, sampling_config: ModelSamplingConfig) -> list[MaskPredictResult]:
        pass

    @abstractmethod
    def get_does_multi_token_prediction(self) -> bool:
        """
        Returns whether the CLM can generate multiple tokens from one mask token
        """
        pass

    @abstractmethod
    def get_mask(self, mask: str) -> str:
        """
        Return a model specific token for the provided generic token
        """
        pass

    def get_input_suffix(self, text: str) -> str:
        return ""

    def get_first_token_number(self) -> int:
        return 0

    @classmethod
    def select_variant_by_name(cls, variant: str) -> MaskPredictModelVariant:
        for v in cls.VARIANTS:
            if v.name.lower() == variant.lower():
                return v
        raise ValueError(
            f"Variant for model of type {cls.__name__} must one of "
            f"{[v.name for v in cls.VARIANTS]}, got {variant}"
        )

    @classmethod
    def get_default_variant(cls) -> MaskPredictModelVariant:
        return cls.VARIANTS[0]

    def __str__(self) -> str:
        return f"MaskPredictModel({self.NAME}-{self.model_variant.name})"
