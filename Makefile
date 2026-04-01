# PI DATA Phase 2 - project automation
# Run make from the folder that contains this Makefile (project root).
# Windows: use GNU Make that ships with Git (full path), e.g.:
#   "C:\Program Files\Git\usr\bin\make.exe" pipeline
# This file avoids paths-with-spaces bugs from MAKEFILE_LIST parsing.

PYTHON ?= python
PIP ?= pip

# PYTHONPATH must be the project root so "import src...." works.
export PYTHONPATH := $(CURDIR)

DATASET ?= eMBB
MODEL ?= rf
ARTIFACT ?= data/models/$(MODEL)/$(DATASET)

.PHONY: help install install-pip compile pipeline train-all \
	train-rf train-xgb train-lr train-et train-mlp train-lgbm \
	predict predict-sample clean-reports

# Plain echo lines only (no nested quotes) - works with cmd.exe + Windows GNU Make
help:
	@echo "Targets (run from project root):"
	@echo "  make install"
	@echo "  make compile"
	@echo "  make pipeline"
	@echo "  make train-all"
	@echo "  make train-rf"
	@echo "  make train-xgb"
	@echo "  make train-lr"
	@echo "  make train-et"
	@echo "  make train-mlp"
	@echo "  make train-lgbm"
	@echo "Optional: DATASET=NAME  MODEL=NAME  example: make train-rf DATASET=eMBB"
	@echo "  make predict"
	@echo "  make predict-sample"
	@echo "  make clean-reports"

install:
	$(PIP) install -r requirements.txt

install-pip: install

compile:
	$(PYTHON) -m py_compile src/train.py src/predict.py src/evaluate.py src/pipeline.py src/data_loader.py src/preprocessing.py src/features.py src/config.py

pipeline:
	$(PYTHON) -c "from src.pipeline import run_mlops_pipeline; run_mlops_pipeline()"

train-all:
	$(PYTHON) -c "from src.train import train_all_models; train_all_models()"

train-rf:
	$(PYTHON) -c "from src.train import train; train('rf', dataset_name='$(DATASET)')"

train-xgb:
	$(PYTHON) -c "from src.train import train; train('xgb', dataset_name='$(DATASET)')"

train-lr:
	$(PYTHON) -c "from src.train import train; train('lr', dataset_name='$(DATASET)')"

train-et:
	$(PYTHON) -c "from src.train import train; train('et', dataset_name='$(DATASET)')"

train-mlp:
	$(PYTHON) -c "from src.train import train; train('mlp', dataset_name='$(DATASET)')"

train-lgbm:
	$(PYTHON) -c "from src.train import train; train('lgbm', dataset_name='$(DATASET)')"

predict:
	$(PYTHON) -m src.predict --artifact "$(ARTIFACT)" --dataset "$(DATASET)"

predict-sample:
	$(PYTHON) -m src.predict --artifact data/models/rf/eMBB --dataset eMBB --rows 3

clean-reports:
	$(PYTHON) -c "import pathlib; p=pathlib.Path('reports/mlops'); p.mkdir(parents=True, exist_ok=True); [f.unlink() for f in p.iterdir() if f.suffix in ('.csv','.json')]; print('cleaned reports/mlops')"
