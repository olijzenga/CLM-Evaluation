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

"""
Returns the number of unique generated infills for each bug and in total.
"""

import json
import fire
import os

from print_results import find_result_jsons, get_results, parse_mask_replacement


def get_perfect_infills_per_task() -> dict[str, str]:
    scriptdir = os.path.dirname(os.path.realpath(__file__))

    correct_files = {}
    humaneval_tasks_dir = os.path.join(scriptdir, 'humaneval-java', 'src', 'main', 'java', 'humaneval', 'correct')
    for task in os.listdir(humaneval_tasks_dir):
        with open(os.path.join(humaneval_tasks_dir, task), 'r') as f:
            correct_files[task] = f.read()

    masked_files = {}
    infilling_tasks_dir = os.path.join(scriptdir, 'single-line-mask-infilling', 'src', 'main', 'java', 'humaneval', 'buggy')
    for task in os.listdir(infilling_tasks_dir):
        with open(os.path.join(infilling_tasks_dir, task), 'r') as f:
            masked_files[task] = f.read()

    result = {}
    for task in correct_files:
        prefix, suffix = masked_files[task].split("<mask>", 1)
        correct_infill = correct_files[task][len(prefix):-len(suffix)]
        result[task] = correct_infill

    return result


perfect = get_perfect_infills_per_task()


def get_nr_correct_fixes(data: dict, n: int) -> int:
    nr_correct = 0
    for results in data['results'].values():
        if 'SUCCESS' in results[:n]:
            nr_correct += 1
    return nr_correct


def get_best_results_for_models(result_data: dict[str, dict], n: int) -> dict[str, dict]:
    results_by_model = {}
    for file_path, data in result_data.items():
        model_name = data['config']['model_name']
        model_variant = data['config']['model_variant']
        slug = f"{model_name}_{model_variant}"

        if slug not in results_by_model:
            results_by_model[slug] = data
            continue

        if get_nr_correct_fixes(data, n) > get_nr_correct_fixes(results_by_model.get(slug), n):
            results_by_model[slug] = data
    return {file_path: data for file_path, data in result_data.items() if data in results_by_model.values()}


def get_nr_unique_results_per_bug(result_data: dict[str, dict], n: int) -> dict[str, int]:
    results_by_bug: dict[str, set] = {}

    for result_file_path, data in result_data.items():
        run_dir = os.path.dirname(result_file_path)
        for task_name in data['results'].keys():
            for i, result in enumerate(get_results(task_name, run_dir, n)):
                if data['results'][task_name][i] != 'SUCCESS':
                    continue

                replacement = parse_mask_replacement(result)
                replacement = replacement.replace(' ', '').replace('\t', '').replace('\n', '')
                perfect_replacement = perfect[task_name].replace(' ', '').replace('\t', '').replace('\n', '').strip()
                if replacement == perfect_replacement:
                    continue

                if task_name in results_by_bug:
                    results_by_bug[task_name].add(replacement)
                else:
                    results_by_bug[task_name] = {replacement}

    return {task_name: len(infills) for task_name, infills in results_by_bug.items()}


def main(dirname: str, n: int = 1):
    result_files_content = {}
    dirs = os.listdir(dirname)
    for dir_ in dirs:
        if not os.path.isdir(os.path.join(dirname, dir_)):
            continue

        result_files = find_result_jsons(os.path.join(dirname, dir_))
        for file_path in result_files:
            with open(file_path, 'r') as f:
                result_files_content[file_path] = json.loads(f.read())

    best_results = get_best_results_for_models(result_files_content, n)
    print("Best configs:")
    for data in best_results.values():
        model_name = data['config']['model_name']
        model_variant = data['config']['model_variant']
        slug = f"{model_name}_{model_variant}"

        print(slug.ljust(25, "."), get_nr_correct_fixes(data, n))

    print("\n")

    nr_results_per_bug = get_nr_unique_results_per_bug(best_results, n)
    print("Nr unique infills per task:")
    for task_name, nr_unique_infills in nr_results_per_bug.items():
        print(task_name.ljust(30, "."), nr_unique_infills)

    print("Total:".ljust(30, "."), sum(nr_results_per_bug.values()))


if __name__ == "__main__":
    fire.Fire(main)

