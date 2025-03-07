#!/bin/bash
# Complete script to install Conda and setup the hexcore environment

# Set up error handling
set -e  # Exit immediately if a command exits with a non-zero status
set -o pipefail  # Return value of a pipeline is the value of the last command to exit with a non-zero status

echo "Starting installation of Conda and hexcore environment setup..."

# Download and install Miniconda
echo "Downloading Miniconda installer..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

echo "Making installer executable..."
chmod +x miniconda.sh

echo "Running Miniconda installer..."
./miniconda.sh -b -p $HOME/miniconda

# Add conda to path and initialize
echo "Adding Conda to PATH and initializing..."
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate
conda init bash

# Create and activate hexcore environment
echo "Creating hexcore environment with Python 3.12..."
conda create -n hexcore python=3.12 -y

echo "Activating hexcore environment..."
source $HOME/miniconda/bin/activate hexcore

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Hugging Face libraries
echo "Installing Hugging Face libraries..."
pip install transformers accelerate bitsandbytes peft

# Install memory optimization libraries
echo "Installing memory optimization libraries..."
pip install flash-attn

# Install knowledge system libraries
echo "Installing knowledge system libraries..."
pip install networkx sentence-transformers
# Install FAISS GPU through conda (corrected method)
conda install -c pytorch faiss-gpu -y

# Install utility libraries
echo "Installing utility libraries..."
pip install psutil numpy tqdm jupyter matplotlib

# Install development tools
echo "Installing development tools..."
pip install black isort mypy pytest

# Verify installation
echo "Verifying installation..."
python -c "import sys; import torch; print(f'Python version: {sys.version.split()[0]}'); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Number of GPUs: {torch.cuda.device_count()}');"

# Verify FAISS installation
python -c "import faiss; print(f'FAISS version: {faiss.__version__}'); print(f'FAISS GPU support available: {faiss.get_num_gpus()}');"

# Export environment
echo "Exporting environment configuration..."
conda env export > environment.yml

echo "Hexcore environment setup complete!"
echo "To activate this environment in the future, run: conda activate hexcore"