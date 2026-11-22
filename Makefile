PY ?= .venv/bin/python
export OMP_NUM_THREADS ?= 1

.PHONY: all setup data test prepare overlap transport design population reproduce report clean

all: data test prepare overlap transport design population reproduce report

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -e ".[test]"

data:
	$(PY) scripts/download_cropglobe.py

test:
	$(PY) -m pytest -q

prepare:        ## maps + domain classifiers (about 3 min on 10 cores)
	OMP_NUM_THREADS=2 $(PY) experiments/prepare.py

overlap:        ## pairwise positivity diagnostic (about 1 min)
	$(PY) experiments/pairwise_overlap.py

transport:      ## zero-overlap transport study (about 15 min)
	$(PY) experiments/transport.py

design:         ## in-target sampling study (about 1 min)
	$(PY) experiments/design.py

population:     ## population-level identification table (about 2 min)
	$(PY) experiments/population.py

reproduce:      ## RF check against published CropGlobe accuracies (about 2 min)
	OMP_NUM_THREADS=10 $(PY) experiments/reproduce.py

report:         ## figures/ and results/summary.md
	$(PY) experiments/report.py

clean:
	rm -rf results/cache results/*.csv results/*.csv.gz results/summary.md figures/*
