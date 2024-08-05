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
Summarizes the results of one or more benchmark run into a CSV file
"""

import csv
import datetime
import io
import json
import os
import fire


def find_result_jsons(dirname: str) -> list[str]:
    dir_files = os.listdir(dirname)

    # dirname is a single workdir
    if "results.json" in dir_files:
        return [os.path.join(dirname, "results.json")]

    # dirname is probably a collection of workdirs
    nr_dirs = 0
    result_files = []
    for dir_file in dir_files:
        dir_path = os.path.join(dirname, dir_file)
        if os.path.isdir(dir_path):
            nr_dirs += 1
            if "results.json" in os.listdir(dir_path):
                result_files.append(os.path.join(dirname, dir_file, "results.json"))
            else:
                print("WARNING: Potentially missing result file in", dir_path)

    print(f"Found {nr_dirs} directories with {len(result_files)} result files")

    return result_files


def parse_mask_replacement(text: str) -> str:
    text = text.split("//Mask replacement:\n")[1]
    lines = text.split("\n")

    def _cleanup_line(line: str) -> str:
        if line.startswith("//"):
            line = line[2:]
        line = line.strip()
        line = line.replace("\t", " ")
        while "  " in line:
            line = line.replace("  ", " ")

        return line

    lines = [_cleanup_line(l) for l in lines]
    mask_replacement = "\n".join(l for l in lines if l != "")
    return mask_replacement.strip()


def get_results(file_name: str, workdir: str, nr_results: int) -> list[str]:
    result = []
    for result_nr in range(nr_results):
        file_path = os.path.join(
            workdir, f"result_{result_nr}", "src", "main", "java", "humaneval", "buggy", file_name
        )
        if os.path.isfile(file_path + ".noresult"):
            # No result so just skip
            continue
        if not os.path.isfile(file_path):
            file_path += ".fail"
        if not os.path.isfile(file_path):
            # Missing result, not throwing errors for bw compatibility
            print("ERROR: missing result", file_path)
            continue

        with open(file_path, "r") as f:
            result.append(f.read())
    return result


def get_nr_unique_results(results: list[str]) -> int:
    mask_replacements = set()
    for result in results:
        mask_replacements.add(parse_mask_replacement(result))
    return len(mask_replacements)


def get_nr_empty_results(results: list[str]) -> int:
    return len([r for r in results if parse_mask_replacement(r) == ""])


def format_results(result_data: list[tuple[str, dict]]) -> str:
    result_data.sort(key=lambda x: x[1]["config"]["completed_at"])

    result_buf = io.StringIO()
    writer = csv.DictWriter(
        result_buf,
        fieldnames=[
            "CLM Name",
            "CLM Variant",
            "Workdir",
            "JobID",
            "Completed At",
            "Mask Time",
            "Total Time",
            "Seed",
            "NrRes",
            "BeamSize",
            "Top_p",
            "Temp",
            "QuantMode",
            "FastTest",
            "NrCores",
            "#UniqRes",
            "#AnySuc",
            "R1 #Suc",
            "R1 #TFail",
            "R1 #CFail",
            "R1 #TCFail",
            "R1 #TTo",
            "R1 #NoRes",
            "Tot #Suc",
            "Tot #TFail",
            "Tot #CFail",
            "Tot #TCFail",
            "Tot #TTo",
            "Tot #NoRes",
            "Nr #EmptyRes",
            "Max VRAM MiB"
        ],
    )
    writer.writeheader()

    for result_file_path, result in result_data:
        result_config = result["config"]

        r1_stats = {
            "SUCCESS": 0,
            "COMPILE_FAIL": 0,
            "TEST_FAIL": 0,
            "TEST_COMPILE_FAIL": 0,
            "TEST_TIMEOUT": 0,
            "NO_RESULT": 0,
        }
        total_stats = {
            "SUCCESS": 0,
            "COMPILE_FAIL": 0,
            "TEST_FAIL": 0,
            "TEST_COMPILE_FAIL": 0,
            "TEST_TIMEOUT": 0,
            "NO_RESULT": 0
        }
        # Number of files for which there was at least one success result
        total_any_success = 0
        nr_empty_results = 0
        nr_unique_results_per_file = []

        for file_name, file_results in result["results"].items():
            r1_stats[file_results[0]] += 1
            for file_result in file_results:
                total_stats[file_result] += 1
            if any(file_result == "SUCCESS" for file_result in file_results):
                total_any_success += 1

            results = get_results(file_name, os.path.dirname(result_file_path), result_config["sampling_config"]["num_return_sequences"])
            nr_unique_results_per_file.append(get_nr_unique_results(results))
            nr_empty_results += get_nr_empty_results(results)

        if len(nr_unique_results_per_file) == 0:
            avg_unique_results = 0.0
        else:
            avg_unique_results = round(sum(nr_unique_results_per_file) / len(nr_unique_results_per_file), 1)

        try:
            workdir_elems = list(result_config["workdir"].split("_"))
            slurm_jid_index = workdir_elems.index("jid")
            slurm_jid = workdir_elems[slurm_jid_index + 1]
        except ValueError:
            slurm_jid = ""

        writer.writerow(
            {
                "CLM Name": result_config["model_name"],
                "CLM Variant": result_config["model_variant"],
                "Workdir": os.path.basename(result_config["workdir"]),
                "JobID": slurm_jid,
                "Completed At": datetime.datetime.fromtimestamp(result_config["completed_at"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Mask Time": "{:.1f}".format(result_config["mask_predict_time"]),
                "Total Time": "{:.1f}".format(result_config["execution_time"]),
                "Seed": result_config["seed"],
                "NrRes": result_config["sampling_config"]["num_return_sequences"],
                "BeamSize": result_config["sampling_config"]["num_beams"],
                "Top_p": result_config["sampling_config"]["top_p"],
                "Temp": result_config["sampling_config"]["temperature"],
                "QuantMode": result_config["load_preferences"]["quantization_mode"],
                "FastTest": "yes" if result_config.get("fast_testing", False) else "no",
                "NrCores": result_config.get("nr_cores")
                or result_config["sampling_config"]["num_return_sequences"],
                "#UniqRes": "{:.1f}".format(avg_unique_results),
                "#AnySuc": total_any_success,
                "R1 #Suc": r1_stats["SUCCESS"],
                "R1 #TFail": r1_stats["TEST_FAIL"],
                "R1 #CFail": r1_stats["COMPILE_FAIL"],
                "R1 #TCFail": r1_stats["TEST_COMPILE_FAIL"],
                "R1 #TTo": r1_stats["TEST_TIMEOUT"],
                "R1 #NoRes": r1_stats["NO_RESULT"],
                "Tot #Suc": total_stats["SUCCESS"],
                "Tot #TFail": total_stats["TEST_FAIL"],
                "Tot #CFail": total_stats["COMPILE_FAIL"],
                "Tot #TCFail": total_stats["TEST_COMPILE_FAIL"],
                "Tot #TTo": total_stats["TEST_TIMEOUT"],
                "Tot #NoRes": total_stats["NO_RESULT"],
                "Nr #EmptyRes": nr_empty_results,
                "Max VRAM MiB": result_config.get("max_memory_usage_mib", -1)
            }
        )

    return result_buf.getvalue()


def main(dirname: str = "workdirs") -> None:
    result_files = find_result_jsons(dirname)

    if len(result_files) == 0:
        print("Error: found no result files")
        exit(1)

    result_data = []
    for result_file in result_files:
        with open(result_file, "r") as f:
            result_data.append((result_file, json.load(f)))

    print(format_results(result_data))


if __name__ == "__main__":
    fire.Fire(main)
