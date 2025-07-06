#TODO: use the same env name specified in your environment.yaml file
CONDA_ENV_NAME = ml_template
#TODO: change the package name with your own
PYTHON_PACKAGE_PATH = src/ml_template
SRC_DIR = src
TEST_DIR = tests
#TODO: configure with your python version
PYTHON_VERSION = 3.11
OS = $(shell uname -s)

define check_cuda_support
	ifeq (($OS), Darwin)
		@echo "MacOS systems have no support for cuda. Install the CPU dependencies instead."
		exit 1
	endif
endef

# ENVIRONMENT MANAGEMENT ###############################################################################################

.PHONY: help install-env install-pack update-env remove-env

help:
	@echo "-------------------------------------------------------------------------"
	@echo "Makefile for $(CONDA_ENV_NAME)"
	@echo "-------------------------------------------------------------------------"
	@echo "Environment:"
	@echo "  install-dep-cu118    It downloads the correct dependencies with a focus for the CUDA 11.8 architecture."
	@echo "  install-dep-cu126    It downloads the correct dependencies with a focus for the CUDA 12.6 architecture."
	@echo "  install-dep-cu128    It downloads the correct dependencies with a focus for the CUDA 12.8 architecture."
	@echo "  install-env-cpu      It downloads the correct dependencies with a focus for the CPU architecture."
	@echo ""
	@echo "Development and Testing:"
	@echo "  lint                 Executes linting and style controls (Ruff/Black/Flake8...)."
	@echo "  format               Applies code formating (Black/Ruff...)."
	@echo "  test                 Executes all tests with pytest."
	@echo "  test-cov             Exectues tests with pytest and shows the coverage report."
	@echo ""
	@echo "Main Execution (ML):"
	@echo "  predict              Executes the inference script (predict.py)."
	@echo "                       Use 'ARGS=\"<hydra_overrides>\"' to provide CLI arguments."
	@echo "                       Example: make predict ARGS=\"ckpt_path=path/to/model.ckpt\""
	@echo ""
	@echo "Cleaning:"
	@echo "  clean                Removes temporary Python files and test's cache."
	@echo "-------------------------------------------------------------------------"

create-env:
	@conda create -n $(CONDA_ENV_NAME) python=3.11
install-dep-cu118:
	$(call check_cuda_support)
	@pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@pip3 install -r requirements.txt
install-dep-cu126:
	$(call check_cuda_support)
	@pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
	@pip3 install -r requirements.txt
install-dep-cu128:
	$(call check_cuda_support)
	@pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
	@pip3 install -r requirements.txt
install-dep-cpu:
	@pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	@pip3 install -r requirements.txt

# DEV AND TESTING ######################################################################################################

.PHONY: lint format test test-cov

lint:
	@echo ">>> Executing linting (Ruff)"
	@conda run -n $(CONDA_ENV_NAME) ruff check $(SRC_DIR) $(TEST_DIR)
	@echo ">>> Checking code formatting (Black)"
	@conda run -n $(CONDA_ENV_NAME) black --check $(SRC_DIR) $(TEST_DIR)

format:
	@echo ">>> Code formatting (Ruff)"
	@conda run -n $(CONDA_ENV_NAME) ruff format $(SRC_DIR) $(TEST_DIR)
	@echo ">>> Code formatting (Black)"
	@conda run -n $(CONDA_ENV_NAME) black $(SRC_DIR) $(TEST_DIR)

test:
	@echo ">>> Executing Test (pytest)"
	@conda run -n $(CONDA_ENV_NAME) pytest $(TEST_DIR)

test-cov:
	@echo ">>> Executing Test with Coverage (pytest)..."
	@conda run -n $(CONDA_ENV_NAME) pytest --cov=$(SRC_DIR) --cov-report=term-missing $(TEST_DIR)


# MAIN EXECUTION #######################################################################################################

.PHONY: train predict

ARGS = ""

predict:
	@echo ">>> Executing Inference (predict.py)..."
	@conda run -n $(CONDA_ENV_NAME) python $(PYTHON_PACKAGE_PATH)/predict.py $(ARGS)


# CLEANING #############################################################################################################

.PHONY: clean clean-outputs

clean:
	@echo ">>> Removing temp Python files and cache..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@rm -rf build/ dist/ *.egg-info/

.DEFAULT_GOAL := help