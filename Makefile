setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

run:
	python run_pipeline.py

test:
	pytest -q
