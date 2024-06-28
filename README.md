# gpt2-prot
Train biological language models at single NT or AA resolution.

## Todo

- [x] Add config recipes for eg. foundation model training, specific protein modelling etc.
- [ ] Docstrings etc.
- [ ] Readme instructions
- [ ] AWS spot instances demo
- [ ] Github actions for publishing the package to pypi
- [ ] Add inference mode

## Installation

  Installation from pypi is on the way 

```bash
micromamba create -f environment.yml  # or conda etc.
micromamba activate gpt2-prot

pip install .  # Basic install
pip install -e ".[dev]"  # Install in editable mode with dev dependencies
pip install ".[test]"  # Install the package and all test dependencies
```

## Usage

### From the CLI

```bash
gpt2-prot -h

gpt2-prot fit --config recipes/cas9_analogues.yml  # Run the demo config for cas9 protein language modelling
```

## Development

### Running pre-commit hooks

```bash
# Install the hooks:
pre-commit install

# Run all the hooks:
pre-commit run --all-files
```

### Running tests

Pytest will find all files with the name "test_*.py" or "*_test.py", run simply by calling `pytest` from the repo root.
