# PNR Graphs Project

## Setup

### Quick Start

Run the setup script to create the conda environment and install all dependencies:

```bash
chmod +x setup.sh
./setup.sh
```

Then activate the environment:

```bash
conda activate pnr-graphs
```


## Notes
- add git repositories that you clone locally into gitignore so we have a clean branch
- Adjust CUDA version in `setup.sh` if you have a different CUDA version
- The script uses `conda run` to avoid needing to manually activate the environment
- Git repositories are cloned in the project directory, b

