install:
	pip install -U pip setuptools wheel
	pip install -e .

lint:
	python -m mypy clip_on_yarn
	python -m pylint clip_on_yarn
	python -m black --check -l 120 clip_on_yarn
	
black:
	python -m black -l 120 clip_on_yarn