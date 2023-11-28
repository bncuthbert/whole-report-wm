# makefile for common repo operations 

.PHONY: all init data

all: # list make recipes w/ descriptions
	@echo -e "make options: \
	\n	'make init': installs required dependencies in the current Python environment\
	\n	'make data': downloads Adam et al data from OSF and performs processing pipeline\
	\n	'make notebooks': generates notebooks from jupytext scripts and vice versa\
	\n 	'make pdf': concatenates .md files in manuscript/ into manuscript/manuscript.pdf"


init: # installs required dependencies in the current Python environment
	@echo "[INSTALLING DEPENDENCIES...]"
	@pip install -r requirements.txt
	@pip install -e .
	@echo "[DEPENDENCIES INSTALLED]"

data: # downloads Adam et al data from OSF and performs processing pipeline
	@echo "[1/4 DOWNLOADING DATA... ]"
	@python src/data/_00_download_data.py
	@echo "[2/4 EXTRACTING DATA FROM .CSV AND .MAT FILES...]"
	@python src/data/_01_extract_data.py
	@echo "[3/4 CLEANING DATAFRAMES...]"
	@python src/data/_02_clean_data.py
	@echo "[4/4 COMPILING DATAFRAMES...]"
	@python src/data/_03_compile_data.py
	@echo "[COMPLETE]"