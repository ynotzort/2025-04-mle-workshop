# MLE Course 2025-04

Here we learn how to do Machine Learning Engineering Stuff.

It is based on this repository by Alexey: https://github.com/alexeygrigorev/ml-engineering-contsructor-workshop


## how to install UV

Just run `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Day 1

- Create a folder named `day_1`
- in the console run `cd day_1` to change the current directory to be day_1
- run `uv init --python 3.10`
- run `uv sync`

### lets install the dependencies
- run `uv add scikit-learn==1.2.2 pandas pyarrow`
- run `uv add --dev jupyter seaborn`
- run `uv add numpy==1.26.4` to fix the issue with sklearn

### lets convert the notebook to a script
- `uv run jupyter nbconvert --to=script notebooks/duration-prediction.ipynb`

### lets make the script nicer

see the git commits for the intermediate steps:
- remove top level statements
- make function parameterized
- introduce argparse
    - make it run via `uv run python duration_prediction/train.py --train-date 2022-01 --val-date 2022-02 --model-save-path models/2022-01.bin`
- add docstrings: eg use autodocstring extension in vscode or chatgpt
- add logging

### makefile

create a file called Makefile and insert content.
now we can run training via `make train`


### add tests
- run `uv add pytest`
- create a tests folder
- in the test folder create a file `test_train.py`
- in the folders `duration_prediction` and `tests` create empty files named `__init__.py`
- run `uv run pytest`

- you can also run the tests via `make tests` thanks to the Makefile

### make it runnable as a module
- move some code to the main.py file
- run it via `uv run python -m duration_prediction.main` instead of `uv run python duration_prediction/main.py`


## Day 2

### create the project
- `uv init --lib --python 3.10 duration_pred_serve`
- add dependencies from day_1: `uv add scikit-learn==1.2.2 numpy==1.26.4`
- add Flask and pytest: `uv add flask pytest`
- add requests: `uv add --dev requests`
- copy over model from day 1
- add loguru `uv add loguru`

### ping example
- can be run via `uv run python src/duration_pred_serve/ping.py`
- then you can look at it via browser `http://127.0.0.1:9696/ping`
- `curl 127.0.0.1:9696/ping`

### implement serve
- run it via `uv run python src/duration_pred_serve/serve.py`
- test it via curl:
`curl -X POST -d '{"PULocationID": "43", "DOLocationID": "238", "trip_distance": 1.16}' -H "Content-Type: application/json" 127.0.0.1:9696/predict`
- or via requests: `uv run python integration-tests/predict-test.py`

- run via Makefile:
    - make run
    - make predict-test
    
### use environment variables
have a look into export
on the python side we use os.getenv

### add logging via loguru
- added via uv add loguru
- use it via `from loguru import logger`

### use docker
add commands to the makefile:
- `make docker_build` and `make docker_run`


### use gunicorn for production
- `uv add gunicorn`
- usually you would run it via `uv run gunicorn --bind=0.0.0.0:9696 src.duration_pred_serve.serve:app`
- but we just update the entrypoint for our dockerfile


### use fly.io for deployment
- create account at fly.io
- install flyctl: `curl -L https://fly.io/install.sh | sh`
- run 
```
export FLYCTL_INSTALL="/home/codespace/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
```
- you can also copy those lines to you ~/.bashrc
- run `flyctl launch`
