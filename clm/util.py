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

import math

import torch


def center_text(text: str, pad_char: str, length: int) -> str:
    assert isinstance(text, str)
    assert isinstance(pad_char, str)
    assert len(pad_char) == 1
    assert isinstance(length, int)

    pad_length = max(0, length - len(text))
    left_padding = pad_char * math.ceil(pad_length / 2)
    right_padding = pad_char * (pad_length // 2)

    return left_padding + text + right_padding


def get_max_line_length(text: str) -> int:
    return max(len(line) for line in text.split("\n"))


def format_java(text: str, tab_size: int = 4) -> str:
    # Extract and remove comments which will be inserted later on
    comment_placeholders = []
    for i, line in enumerate(text.split("\n")):
        if line.strip().startswith("//"):
            placeholder = f"[comment_{i}];"
            comment_placeholders.append((placeholder, line.strip()))
            text = text.replace(line.strip(), placeholder, 1)

    # Remove existing indentation so that we can also accept already formatted code
    while "\n " in text:
        text = text.replace("\n ", "\n")  # Remove leading spaces
    text = text.replace("\n", "").replace("\t", " ")
    for c in (";", "{", "}"):  # Remove spaces after these characters
        while f"{c} " in text:
            text = text.replace(f"{c} ", c)

    result = ""
    indent = ""

    for i, c in enumerate(text):
        result += c

        if c == "{":
            indent += " " * tab_size
            result += "\n" + indent
        elif c == ";":
            result += "\n" + indent
        elif c == "}":
            indent = indent[:-tab_size]
            result = result[: -tab_size - 1] + "}"  # Remove curly brace and put it back with less indent

            # Deal with 'else' block
            remainder = text[i + 1 :]
            if remainder.strip().startswith("else"):
                if not remainder.startswith(" "):
                    result += " "
                continue

            result += "\n" + indent

    # Re-insert comments
    for placeholder, comment in comment_placeholders:
        result = result.replace(placeholder, comment, 1)

    return result


def get_torch_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(device_name)


def get_indent(line: str) -> str:
    """
    Returns the indentation characters at the start of the line
    """
    result = ""
    for c in line:
        if c in " \t":
            result += c
        else:
            break
    return result


def remove_prefix_ignoring_whitespaces(prefix: str, subject: str) -> str:
    prefix = prefix.replace(" ", "").replace("\t", "").replace("\n", "")
    for i, c in enumerate(subject):
        if len(prefix) == 0:
            return subject[i:]

        if c in " \t\n":
            continue

        if prefix[0] != c:
            return subject

        prefix = prefix[1:]

    return ""


def remove_suffix_ignoring_whitespaces(suffix: str, subject: str) -> str:
    return remove_prefix_ignoring_whitespaces(suffix[::-1], subject[::-1])[::-1]
