
.PHONY: run
run:
	export MODEL_PATH="./models/2022-01.bin"; \
	uv run python src/duration_pred_serve/serve.py

.PHONY: predict-test
predict-test:
	uv run python integration-tests/predict-test.py
