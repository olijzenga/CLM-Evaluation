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

import os
import random

LOC_FILE = "../humaneval-java/src/main/java/humaneval/humaneval_loc.txt"
CORRECT_DIR = "../humaneval-java/src/main/java/humaneval/correct"
OUT_DIR = "src/main/java/humaneval/buggy"
ANSWERS_FILE = "answers.txt"
MASK = "<mask>"


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


def load_files() -> dict[str, str]:
    """Returns tuples of name and file content"""
    file_names = os.listdir(CORRECT_DIR)

    result = {}
    for file_name in file_names:
        with open(os.path.join(CORRECT_DIR, file_name), "r") as f:
            result[file_name[:-5]] = f.read()

    return result


def load_buggy_locations() -> dict[str, tuple[int, int]]:
    """Returns tuples of name and start and end of bug location"""
    with open(LOC_FILE, "r") as f:
        lines = [l for l in f.readlines() if l.strip() != ""]

    result = {}
    for line in lines:
        [name, location_str] = line.split(" ", 1)
        location_str = location_str.strip()
        [start_str, end_str] = location_str.split("-", 1)

        result[name] = (int(start_str), int(end_str))
    return result


def mask_random_line(name: str, code: str, start_line: int, end_line: int) -> tuple[str, str]:
    """Returns tuple of new code and masked line"""
    lines = code.split("\n")

    assert end_line - 1 <= len(lines), (name, code, start_line, end_line)  # end line needs to exist

    for _ in range(10):
        masked_line_i = random.randint(start_line, end_line) - 1
        line = lines[masked_line_i]
        line_indent = get_indent(line)

        if len(line.strip()) <= 3:
            continue

        print(f"[{name}]".ljust(40, "."), line.lstrip())
        lines[masked_line_i] = line_indent + MASK

        return "\n".join(lines), line.lstrip()

    raise Exception(f"Failed to generate good mask for {(name, code, start_line, end_line)}")


def main():
    print("WARNING: This script re-generates the benchmark suite, overwriting the existing one.")
    print("Do you want to proceed? [y/n] ", end="")
    answer = input()

    if answer.strip().lower() != 'y':
        print("Aborted")
        return

    files = load_files()
    buggy_locations = load_buggy_locations()

    with open(ANSWERS_FILE, 'w') as answers_file:
        for name, content in files.items():
            (start, end) = buggy_locations[name]
            masked_content, answer = mask_random_line(name, content, start, end)

            # Replace humaneval.correct package name with humaneval.buggy
            masked_content = masked_content.replace('humaneval.correct', 'humaneval.buggy')

            with open(os.path.join(OUT_DIR, name + ".java"), "w") as f:
                f.write(masked_content)

            answers_file.write(f'{name} {answer}\n')

    print(f"Generated {len(files)} files")


if __name__ == "__main__":
    main()
