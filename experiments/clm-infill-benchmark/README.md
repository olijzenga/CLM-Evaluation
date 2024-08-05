# HumanEval-Java Benchmark

This folder contains the experimental setup for the evaluaton of code infilling capabilities of CLMs. The remainder
of this README explains the purpose of each file in this directory.

## humaneval-java-infill.zip
This archive contains the benchmark suite for single-line infilling based on HumanEval-Java. The benchmark tasks
are archived to avoid web crawlers from collecting these files as training data for CLMs. The `src` directory 
of the archive contains 163 files which each correspond to an infill task. The `test` directory contains the respective
tests of each source file. 

## generate_prompts.py
This is the script used to generate the humaneval-java-infill benchmark. This script should be executed inside
a copy of humaneval-java. It selects a random buggy location in the buggy version of each file, and replaces
the respective line in the correct version of the file with `<mask>`.

## benchmark_clm_infilling.py
This script applies the humaneval-java-infill benchmark to a CLM supported by the `clm` package. The script is used
as follows:

```shell
python benchmark_clm_infilling.py [clm name]

# Usage details
python benchmark_clm_infilling.py --help
```

Results will be written to the `workdirs` directory (automatically generated) which contains all intermediate 
results and a `results.json` file which contains the evaluation summary and metadata. Use the help command
for usage details.

## print_results.py

Prints a CSV summary of the results of either a single benchmark output directory, or of a directory containing
multiple output directories.

```shell
# A specific workdir
python print_results.py workdirs/[some specific workdir]

# All workdirs
python print_results.py workdirs
```

## get_infill_uniqueness.py

Prints the number of unique infills per task, and the overall average number of unique infills for a specific working
directory.

## get_overlap_analysis.py

Analyzes multiple workdirs to find the overlap between infill correctness results of different CLMs. Also prints
which bugs are not fixed by any CLM.

## get_unique_fixes.py

