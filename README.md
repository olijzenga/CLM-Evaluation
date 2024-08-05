# CLM Evaluation

This repository contains the code for performing a comparative analysis of 20 code language models (CLM). This work was part of my master thesis, and everything is published
under an MIT license, so feel free to use it.

## Structure

The `clm` folder contains the code that unifies the mask prediction capabilities of many CLMs under a single interface. The `experiments` folder contains the infilling benchmark
based on HumanEval-Java, and the scripts for performing the evaluation.

## Requirements

All code has been developed and run on Linux. Some minor adjustments might be required to run it on MacOS or Windows.

- A recent version of the JDK
- Maven
- Python 3.9 or greater
- CUDA (for running CLMs on Nvidia GPUs)

## Installation

Dependencies for the `clm` package are managed through [Poetry](https://github.com/python-poetry/poetry), and can be installed via
Poetry, or via Pip:

```shell
# Using Pip
pip install .

# Using Pip (development mode)
pip install -e .

# Using Poetry
poetry install  # Install dependencies
poetry shell   # Activate virtual environment
```

## Usage

The CLM package can be either used through the command line, or the API.

### Mask Prediction CLI

The example below shows a simple mask predict command. The input file should contain a `<mask>`, which is the universal mask token
which is translated to the appropriate mask prediction format for each CLM. For a list of available CLMs and their variants, either provide an incorrect
CLM name and it will provide the available values, or check the CLM names of `MaskPredictModel` subclasses in the `clm/clms` folder.

```shell
# Example usages
python mask_predict.py codet5 --model-variant=large ./inputs/abs_expr.py
python mask_predict.py codet5 --model-variant=large ./inputs/abs_expr.py --nr-beams=5
python mask_predict.py codet5 --model-variant=large ./inputs/abs_expr.py --top-p=0.5 temperature=1
python mask_predict.py codet5 --model-variant=large ./inputs/abs_expr.py --quantization-mode=8bit

# Usage info
python mask_predict.py --help
```

### Mask Prediction API

The mask prediction API provides similar functionality compared to the CLI but in web API form.
This is used for executing mask prediction from Java code. The API is started as follows:

```shell
flask --app clm.api.api run  # Standard mode
flask --app clm.api.api --debug run  # Debug/development mode
```

Flask development mode also enables automatic reloading of the API when code is changed.

#### API Endpoints

Currently, there is only one API endpoint.

##### POST /mask_predict
* Path parameters: none
* Query parameters: none
* Request body:

```json
{
  "text": "def hello_world(): <mask>",  // Input with <mask> token
  "model_name": "unixcoder",  // Name of the CLM
  "model_variant": null,  // CLM variant, available options differ per CLM
  "nr_results": 10  // Number of mask predictions to generate, defaults to 10
}
```
