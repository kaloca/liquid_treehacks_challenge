IS_CONDA_ENV=$(if $(CONDA_PREFIX),true,false)
IS_VENV=$(if $(VIRTUAL_ENV),true,false)

PYTHON_MIN_VERSION := 3.10
PYTHON_OK := $(shell python3 -c 'import sys; exit(0) if sys.version_info >= (3,10) else exit(1)' >/dev/null 2>&1 && echo yes || echo no)

# -----------------------------------------------------------------------------
# checks
# -----------------------------------------------------------------------------
_check_ccache:
	@command -v ccache >/dev/null 2>&1 \
		&& { echo "ccache found at: $(shell command -v ccache) $(shell ccache --version | head -1)"; } \
		|| { echo "Ccache is missing, without it the build is slow, follow the instructions https://ccache.dev/download.html" ; }

_check_ninja:
	@command -v ninja >/dev/null 2>&1 \
		&& { echo "ninja found at: $(shell command -v ninja) $(shell ninja --version)"; } \
		|| { echo "Nija is missing, follow the instructions https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages" ; exit 1; }

# -----------------------------------------------------------------------------
_check_rust:
	@command -v rustc >/dev/null 2>&1 \
		&& { echo "rustc found at: $(shell command -v rustc) $(shell rustc --version)"; } \
		|| { echo "Rust is missing, follow the instructions https://www.rust-lang.org/tools/install" ; exit 1; }

# ------------------------------------------------------------------------------
# g++ check
# ------------------------------------------------------------------------------

_check_gpp_version:
	@echo "Checking g++ version"
	@GPP_VERSION=`g++ --version | grep -oP '([0-9]+\.)+[0-9]+' | head -n 1`; \
	MAJOR=$(echo "$$GPP_VERSION" | cut -d. -f1 | tr -d ' '); \
	echo "g++ version is $$GPP_VERSION"; \
	echo "GPP_VERSION is $$GPP_VERSION, if it is less than 13, please update. A quick way is to use conda forge: conda install -c conda-forge gxx"; \

# ------------------------------------------------------------------------------
# check tools
# ------------------------------------------------------------------------------

check-tools: \
	_check_env_enabled \
	_check_ccache \
	_check_ninja \
	_check_rust

check-tools-android: \
	check-tools \
	_check_java \
	_check_android_ndk \
	_check_android_home

# ------------------------------------------------------------------------------
# Python environment setup
# ------------------------------------------------------------------------------

define _CHECK_ENV_MESSAGE
This target has to use conda or virtual environment, did you forget to activate one?

If you have no environment yet, use one of the following commands to create one, it has to be python 3.10+.

conda:

	conda create -yn treehacks python=3.11.0
	conda activate treehacks

venv:

	python -m venv .env --python python3.11
	# or use a default python
	python -m venv .env
	. .env/bin/activate

uv:

	uv venv --seed --python 3.10
	# or use a default python
	uv venv --seed
	. .venv/bin/activate

endef
export _CHECK_ENV_MESSAGE

_check_env_enabled:
ifeq ($(CI), true)
	@echo "Running in CI $(shell python --version)"
else ifeq ($(CONDA_DEFAULT_ENV), base)
	@echo "Not doing anything in the base conda environment"
	@echo "$$_CHECK_ENV_MESSAGE"
	exit 1
else ifeq ($(IS_CONDA_ENV), true)
	@echo "Using conda environment: $(CONDA_PREFIX) $(shell python --version)"
else ifeq ($(IS_VENV), true)
	@echo "Using venv: $(VIRTUAL_ENV) $(shell python --version)"
else
	@echo "$$_CHECK_ENV_MESSAGE"
	exit 1
endif
ifeq ($(PYTHON_OK), no)
	@echo "Python version is less than $(PYTHON_MIN_VERSION)"
	@echo "$$_CHECK_ENV_MESSAGE"
	exit 1
endif

# ------------------------------------------------------------------------------
# check git lfs
# ------------------------------------------------------------------------------

_check_git_lfs:
	@command -v git-lfs >/dev/null 2>&1 \
		&& { echo "git-lfs found at: $(shell command -v git-lfs) $(shell git-lfs --version)"; } \
		|| { echo "git-lfs is missing, follow the instructions https://git-lfs.github.com/"; exit 1; }

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------

setup-submodules: _check_env_enabled
	# the buck2 daemon may still exist and won't be functioning if it is a re-build
	# act in advance and kill any existing instance
	pkill buck2 || true

	# first clean then update
	git submodule foreach --recursive git clean -ffdx
	git submodule foreach --recursive git reset --hard

	# re-apply the patches in the cmake stage but keep the submodules fresh
	# github actions won't clean sub-repos https://github.com/actions/checkout/issues/358
	git submodule update --init --recursive
	git submodule sync --recursive
setup-cmake: _check_env_enabled
	python -m pip install -r requirements-cmake.txt
setup-python-deps: _check_env_enabled
	# speed up the progress
	@command -v uv >/dev/null 2>&1 \
		&& { uv pip install -r requirements.txt --no-progress --index-strategy unsafe-first-match; } \
		|| { python -m pip install -r requirements.txt; }
setup-executorch: setup-python-deps 
	cd vendor/executorch && ./install_executorch.sh --pybind xnnpack
executable-runner: _check_git_lfs
	git lfs fetch --all
	chmod +x runner/linux/pte_runner
	chmod +x runner/android-arm64-v8a/pte_runner
setup: _check_env_enabled setup-submodules setup-executorch executable-runner
setup-compile: _check_env_enabled setup-submodules setup-cmake setup-python-deps
