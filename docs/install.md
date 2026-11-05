# Installation

## Requirements

- Python 3.10 or newer
- pip
- Git

## Clone the repository

Clone the project repository:

```bash
git clone https://github.com/nabdullaev/kfactors.git
cd kfactors
```

## Create a virtual environment

Create an isolated Python environment:

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux and macOS:**

```bash
source .venv/bin/activate
```

## Install the package

Install `kfactors` in editable mode:

```bash
python -m pip install -e .
```

## Install documentation dependencies

To build or preview the documentation, install the documentation dependencies:

```bash
python -m pip install -r docs/requirements.txt
```

## Verify the installation

Check that Python can import the package:

```bash
python -c "import kfactors; print('kfactors installed successfully')"
```