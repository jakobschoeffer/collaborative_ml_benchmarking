# Untapping Analytical Synergies in Industrial SME Ecosystems: An Empirical Evaluation of Federated Machine Learning

## Usage
- please read this README carefully
- start setting up the project as described in Project Setup.
- see File Contents for an overview over the project
- you will want to run fed_ml_horizontal.pipeline.run_pipeline
  - if you are using VSCode, the needed launch specification is included in the repo in .vscode/launch.json. Simply click on "Run and Debug" in the side bar left and on the "play" button on top to run the pipeline in debug mode.
  - debug mode is slow, for running the code in the background in the server, run "poetry shell" followed by "nohup python -m fed_ml_horizontal.pipeline.run_pipeline &" in the command line
- after running the code once, you don't want to redownload the data. Hence, you can comment out the first step in fed_ml_horizontal/config/project.yml
- scenarios for the benchmarking are specified in fed_ml_horizontal/config/scenarios.yml. You can remove and add scenarios here. An excel file for creating nice scenario specifications easily is provided in tesis/share_...
- in case you want to use a unified test data set over all clients and settings, choose "unified_test_dataset: true" in the project.yml in the config file

## File Contents
- fed_ml_horizontal: all code for horizontal federated learning benchmark
  - benchmarking: benchmarking code
    - all_data_model: all data model
    - data_splitting: code for splitting up data for simulating clients
    - federated_learning: federated learning
    - model: contains the tensorflow model that is used in all scenarios and settings
    - one_model_per_client: one model per client
    - plotting: plotting and saving figs and csvs
    - run_scenarios: code for executing the benchmarking steps
    - summaries: code for calculating and exporting summaries such as performance and welfare gains
    - tf_utils: code for creating tensorflow datasets
  - config: configuration for pipeline (whole project), steps and scenarios
  - etl.load_data: code for loading data
  - log: automatically created for logging
  - outputs: automatically created for outputs. Outputs are stored here.
  - pipeline: code to run pipeline steps
  - tmp: data is being stored here. only created when running the pipeline
  - util: utils for setting up logging and reading config
- notebooks
  - boxplot_transformer: adjusts angle of x-axis ticks after plot was created
  - table_transformer: transposes csvs and creates tex files for thesis (included in normal code, just for old csvs)
- presentations
  - final_presentation: final presentation of master's thesis as PowerPoint and pdf
  - interim_presentation: interim presentation of master's thesis as pdf
- thesis: tex code, excel file for share calculation, thesis pdf


## Project Setup
- clone this repo
- Install python using pyenv
  - on linux (or WSL II):
    - run "curl https://pyenv.run | bash" to install pyenv
    - as recommended in the installation output, add the three recommended lines as explained in console output to "~/.profile" e.g. using nano. The tree lines are:
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
