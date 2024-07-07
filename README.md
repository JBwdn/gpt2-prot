# gpt2-prot
Train biological language models at single NT or AA resolution.

## Roadmap

- [ ] Readme instructions
- [ ] AWS spot instances demo
- [ ] Update recipe configs with new inference flags
- [x] Add inference mode
- [x] Add config recipes for eg. foundation model training, specific protein modelling etc.
- [x] Github actions for publishing the package to pypi
- [x] Docstrings etc.

## Installation

```bash
pip install gpt2_prot
```

### From source

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

# Run the demo config for cas9 protein language modelling
# Since this uses Lightning you can overwrite parameters from the config using the command line
gpt2-prot fit --config recipes/cas9_analogues.yml --max_epochs 10  
```

## Development

### Running pre-commit hooks

```bash
# Install the hooks:
pre-commit install

# Run all the hooks:
pre-commit run --all-files

# Run unit tests:
pytest
```
