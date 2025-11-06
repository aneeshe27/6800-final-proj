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

### Manual Setup

If you prefer to set up manually:

1. Create conda environment:
   ```bash
   conda create -y -n pnr-graphs python=3.10
   conda activate pnr-graphs
   ```

2. Install PyTorch (CUDA 12.1):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. Install core libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch Geometric:
   ```bash
   pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
   ```

5. Clone and install GroundingDINO:
   ```bash
   git clone https://github.com/IDEA-Research/GroundingDINO.git
   pip install -e GroundingDINO
   ```

6. Clone and install SAM2:
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   pip install -e ./sam2
   ```

## Notes

- Adjust CUDA version in `setup.sh` if you have a different CUDA version
- The script uses `conda run` to avoid needing to manually activate the environment
- Git repositories are cloned in the project directory

