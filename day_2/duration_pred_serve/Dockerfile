FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.6.12 /uv /uvx /bin/

# copy over all files
# ADD . /app/

# add the files pyproject.toml uv.lock README.md and the folder src
# to the directory /app in the docker image
ADD pyproject.toml uv.lock README.md src/ /app/
WORKDIR /app

RUN uv sync --frozen

# copy over the model
ADD models/2022-01.bin 2022-01.bin

# set the environment variables
ENV MODEL_PATH 2022-01.bin
ENV VERSION 2022-01-v01
ENV LOGURU_LEVEL INFO

EXPOSE 9696
ENTRYPOINT [ "uv", \
             "run", \
             "gunicorn", \
             "--bind=0.0.0.0:9696", \
             "duration_pred_serve.serve:app" ]
