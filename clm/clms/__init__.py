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

from .codebert import CodeBERT, CodeBERTMaskPredictModel
from .codet5 import CodeT5, CodeT5MaskPredictModel
from .codet5p import CodeT5p, CodeT5pMaskPredictModel
from .plbart import PLBART, PLBARTMaskPredictModel
from .incoder import InCoder, InCoderMaskPredictModel
from .codegen import CodeGen, CodeGenMaskPredictModel
from .codegen2 import CodeGen2, CodeGen2MaskPredictModel
from .codegen25 import CodeGen25, CodeGen25MaskPredictModel
from .gpt_neo import GPTNeoMaskPredictModel
from .unixcoder import UniXCoder, UniXcoderMaskPredictModel
from .codegeex2 import CodeGeeX2, CodeGeeX2MaskPredictModel
from .refact import Refact, RefactMaskPredictModel
from .starcoder import StarCoder, StarCoderMaskPredictModel
from .santacoder import SantaCoder, SantaCoderMaskPredictModel
from .codellama import CodeLlama, CodeLlamaMaskPredictModel
from .codeshell import CodeShell, CodeShellMaskPredictModel
from .model_factory import MaskPredictModelFactory, ALL_MODELS, ALL_MODELS_BY_NAME
