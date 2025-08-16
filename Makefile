IMAGE_NAME=pot-experiments

.PHONY: build test

build:
	docker build -t $(IMAGE_NAME) .

test: build
	docker run --rm $(IMAGE_NAME) pytest -q
