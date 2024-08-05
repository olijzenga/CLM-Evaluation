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

from transformers import StoppingCriteria, StoppingCriteriaList


class BaseStoppingCriteria(StoppingCriteria):
    def to_list(self) -> StoppingCriteriaList:
        return StoppingCriteriaList([self])


class StoppingCriteriaContainsToken(BaseStoppingCriteria):
    def __init__(self, stops: list, encounters=1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_count = 0
        for stop in self.stops:
            stop_count = (stop == input_ids[0]).sum().item()

        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class StoppingCriteriaContainsString(BaseStoppingCriteria):
    def __init__(self, stops: list, tokenizer, after: str | None = None, stop_after: bool = False, encounters=1):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
        self.after = after
        self.stop_after = stop_after
        self.encounters = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        ids = input_ids[0]
        if self.stop_after:
            ids = ids[:-1]
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        if self.after is not None:
            text = text.split(self.after)[-1]

        stop_count = 0
        for stop in self.stops:
            stop_count = max(stop_count, text.count(stop))

        if stop_count >= self.encounters:
            return True
        return False
