#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=5000
#SBATCH --account=torch_pr_292_courant
#SBATCH --time=00:40:00

module load anaconda3/2025.06
source $(conda info --base)/etc/profile.d/conda.sh

conda clean --packages --tarballs -y
conda remove -y --all -n constellaration || true
conda create -y -n constellaration python=3.12 ipykernel
conda activate constellaration

conda install -y -c nvidia -c conda-forge \
  cuda-toolkit=12.2 \
  cudnn \
  nccl \
  cuda-nvcc=12.2

conda install -y anaconda::libnetcdf
conda install -y -c conda-forge \
  gcc_linux-64=13 gxx_linux-64=13 icu ninja cmake pybind11=2.11

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_PREFIX/bin:$PATH

pip install -U "jax[cuda12]"==0.6.0
pip install --no-build-isolation --verbose booz-xform
pip install datasets pyevtk

mkdir -p code
cd code

git clone https://github.com/lankef/quadcoil.git
git clone https://github.com/hiddenSymmetries/simsopt.git
git clone https://github.com/proximafusion/constellaration.git

cd quadcoil
pip install -e .

cd ../simsopt
pip install -e .
pip install --no-build-isolation virtual-casing

cd ../constellaration
pip install -e .

pip freeze > pip_freeze.txt