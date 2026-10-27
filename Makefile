PY ?= .venv/bin/python

.PHONY: setup data simulate growth abtest fraud fraud-sample analytics all test notebooks dashboard clean

setup:            ## create venv and install everything
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev,app,data]"
	$(PY) -m ipykernel install --sys-prefix --name upi-growth-fraud --display-name "Python (upi-growth-fraud)"

data:             ## download the IEEE-CIS fraud dataset (needs ~/.kaggle/kaggle.json)
	$(PY) -m upi.cli download

simulate:         ## generate the 100K-user UPI platform
	$(PY) -m upi.cli simulate

growth:           ## retention, funnel, CLV, segmentation
	$(PY) -m upi.cli growth

abtest:           ## analyse the three growth experiments
	$(PY) -m upi.cli abtest

fraud:            ## train + evaluate the fraud model on the full Kaggle data
	$(PY) -m upi.cli fraud

fraud-sample:     ## quick fraud run on the first 100K rows
	$(PY) -m upi.cli fraud --nrows 100000

analytics: simulate growth abtest

all: analytics fraud

test:
	$(PY) -m pytest -q

notebooks:        ## execute the analytics notebooks in place (04 needs the Kaggle data: run it after `make data`)
	.venv/bin/jupyter-nbconvert --to notebook --execute --inplace notebooks/0[1-3]*.ipynb

dashboard:
	$(PY) -m streamlit run app/dashboard.py

clean:
	rm -rf data/simulated models reports/figures/*.png
