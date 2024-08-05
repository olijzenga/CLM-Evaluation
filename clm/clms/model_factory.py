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

from clm.clms import (
    UniXcoderMaskPredictModel,
    CodeBERTMaskPredictModel,
    CodeT5MaskPredictModel,
    CodeT5pMaskPredictModel,
    CodeGenMaskPredictModel,
    CodeGen2MaskPredictModel,
    CodeGen25MaskPredictModel,
    GPTNeoMaskPredictModel,
    InCoderMaskPredictModel,
    PLBARTMaskPredictModel,
    RefactMaskPredictModel,
    StarCoderMaskPredictModel,
    SantaCoderMaskPredictModel,
    CodeLlamaMaskPredictModel,
    CodeShellMaskPredictModel
)
from clm.model import MaskPredictModel, ModelLoadPreferences, ModelLoadConfig

ALL_MODELS = [
    UniXcoderMaskPredictModel,
    CodeBERTMaskPredictModel,
    CodeT5MaskPredictModel,
    CodeT5pMaskPredictModel,
    CodeGenMaskPredictModel,
    CodeGen2MaskPredictModel,
    CodeGen25MaskPredictModel,
    GPTNeoMaskPredictModel,
    InCoderMaskPredictModel,
    PLBARTMaskPredictModel,
    RefactMaskPredictModel,
    StarCoderMaskPredictModel,
    SantaCoderMaskPredictModel,
    CodeLlamaMaskPredictModel,
    CodeShellMaskPredictModel
]

ALL_MODELS_BY_NAME = {model.NAME: model for model in ALL_MODELS}


class MaskPredictModelFactory:
    def __init__(self, load_preferences: ModelLoadPreferences):
        assert isinstance(load_preferences, ModelLoadPreferences)

        self._load_preferences: ModelLoadPreferences = load_preferences
        self._model_cache: list[MaskPredictModel] = []

    def get_model(self, model_name: str, model_variant_name: str | None) -> MaskPredictModel:
        model_cls = ALL_MODELS_BY_NAME.get(model_name)
        if model_cls is None:
            raise ValueError(f"Unkown model {model_name}, must be one of {ALL_MODELS_BY_NAME.keys()}")

        if model_variant_name is not None:
            model_variant = model_cls.select_variant_by_name(model_variant_name)
        else:
            model_variant = model_cls.VARIANTS[0]

        for model in self._model_cache:
            if not isinstance(model, model_cls):
                continue

            if model.model_variant.name == model_variant.name:
                return model

        load_config = ModelLoadConfig.create_for_preferences_and_model(self._load_preferences, model_cls)
        model = model_cls(load_config, model_variant)
        self._model_cache.append(model)

        return model
