# collaborative_ml_benchmarking
Federated Machine Learning

## Usage
- you will want to run fed_ml_horizontal.pipeline.run_pipeline
  - if you are using VSCode, the needed launch specification is included in the repo in .vscode/launch.json. Simply click on "Run and Debug" in the side bar left and on the "play" button on top to run the pipeline in debug mode.
- after running it once, you don't want to redownload the data. Hence, you can comment out the first step in fed_ml_horizontal/config/project.yml
- scenarios for the benchmarking are specified in fed_ml_horizontal/config/scenarios.yml. You can remove and add scenarios here.

## Issues
- currently, we are being kicked out from the server when using too many resources. Hence, number of epochs, number of runs etc. is set to a minimum.
- GPU does not work yet.

## File Contents
- fed_ml_horizontal: all code for horizontal federated learning benchmark
  - config: configuration for pipeline (whole project), steps and scenarios
  - pipeline: code to run pipeline steps
  - etl.load_data: code for loading data
  - benchmarking: benchmarking code
    - run_scenarios: code for executing the benchmarking steps
    - data_splitting: code for splitting up data for simulating clients
    - tf_utils: code for creating tensorflow datasets
    - plotting: plotting and saving figs and csvs
    - model: contains the tensorflow model that is used in all scenarios and settings
    - federated_learning: federated learning
    - baseline_model: baseline model
    - one_model_per_client: one model per client
  - utils: utils for setting up logging and reading config
  - tmp: data is being stored here. only created when running the pipeline
  - outputs: outputs are stored here
  - log: log files are stored here


## Project Setup
- clone this repo
- Install python using pyenv
  - on linux (or WSL II):
    - run "curl https://pyenv.run | bash" to install pyenv
    - as recommended in the installation output, add the three recommended lines as explained in console output to ~/.profile e.g. using e.g. nano "~/.profile". The tree lines are:
        ```
        export PYENV_ROOT="$HOME/.pyenv"
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init --path)"
        ```
    - run "source ~/.profile" to reload bashrc
    - before installing python, install build dependencies (build-dep) as specified in <https://devguide.python.org/setup/#linux>
      - on debian or ubuntu, do the following:
        - add "deb-src http://archive.ubuntu.com/ubuntu/ bionic main" to "/etc/apt/sources.list", e.g. by using "nano /etc/apt/sources.list"
        - run "sudo apt-get update"
        - run "sudo apt-get build-dep python3" (or maybe sudo apt-get build-dep python3.6)
    - run "pyenv install 3.8.6" to install python 3.8.6
    - run "pyenv local 3.8.6" to use python 3.8.6
- Install poetry (<https://python-poetry.org/docs/basic-usage/>) (linux: curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -)
  - as explained in the installation output, add poetry to the path and make it available to the command line.
  - run "poetry config virtualenvs.in-project true" to set default behavior for venv creation to in-project
  - in the directory of this repo, run "poetry install" to install all requirements from poetry.lock (or pyproject.toml, if the .lock file does not exist)
    - poetry lists all requirements in the poetry pyproject.toml file. The requirements are then resolved, creating the poetry.lock file.
  - the .venv is now created in the project directory. You want to set this .venv as your python interpreter.
- if you want to run some code from the shell, run "poetry shell" to activate the .venv
- run "pre-commit install" to install the pre-commit hooks that ensure clean code
