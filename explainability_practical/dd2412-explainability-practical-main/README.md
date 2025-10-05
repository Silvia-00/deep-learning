# DD2412 Explainability practical

Deep learning models are becoming better and better at making predictions.
As researchers, regulators, and users, we are also interested in asking additional questions. 
Namely, we would like to _explain_ a decision in terms of the input.
Where in an image is a model focusing on? 
What cues is the prediction based on? does it match our expectation?
Can the model be trusted?

In the [first part](./ExplanationMethods.ipynb) of this practical,
we will explore popular methods for explaining decisions made by image classifiers. With a working implementation, we will qualitatively compare explanations and we will quantitatively evaluate their correctness.

In the [second part](./ModelBias.ipynb) of this practical, we will see 
a practical use of explanations to identify a faulty classifier that was trained on biased data.

## Setup
Install [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
and verify the driver version:
```bash
cat /proc/driver/nvidia/version
# NVRM version: NVIDIA UNIX x86_64 Kernel Module  470.86  Tue Oct 26 21:55:45 UTC 2021
# GCC version:  gcc version 7.5.0 (Ubuntu 7.5.0-3ubuntu1~16.04) 
```

Create a conda environment:
```bash
ENV_NAME='dd2412'

mamba create -y -n "${ENV_NAME}" -c pytorch -c conda-forge \
  python black isort \
  numpy pandas matplotlib seaborn scikit-learn tabulate tqdm \
  jupyterlab ipywidgets jupyterlab_code_formatter jupyter_console \
  pytorch cudatoolkit-dev cudnn

conda activate "${ENV_NAME}"

mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib/"' > "${CONDA_PREFIX}/etc/conda/activate.d/ld_library_path.sh"
echo 'export BETTER_EXCEPTIONS=1' > "${CONDA_PREFIX}/etc/conda/activate.d/better_exceptions.sh"

conda deactivate
conda activate "${ENV_NAME}"

# See details at https://github.com/google/jax#installation
pip install "jax[cuda]" -f 'https://storage.googleapis.com/jax-releases/jax_cuda_releases.html'

pip install \
  flax optax \
  'git+https://github.com/n2cholas/jax-resnet.git' \
  tensorflow-cpu tensorflow-datasets \
  better_exceptions
```

Checks:
```bash
conda list | grep -E 'torch|cudatoolkit|cudnn'
# cudatoolkit     11.6.0                hecad31d_10                  conda-forge
# cudatoolkit-dev 11.4.0                h5764c6d_5                   conda-forge
# cudnn           8.4.1.50              hed8a83a_0                   conda-forge
# jaxlib          0.3.22+cuda11.cudnn82 pypi_0                       pypi
# pytorch         1.12.1                py3.10_cuda11.6_cudnn8.3.2_0 pytorch

which nvcc
# ~/miniconda3/envs/dd2412/bin/nvcc

nvcc --version
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2021 NVIDIA Corporation
# Built on Wed_Jun__2_19:15:15_PDT_2021
# Cuda compilation tools, release 11.4, V11.4.48
# Build cuda_11.4.r11.4/compiler.30033411_0

grep CUDNN_MAJOR -A2 -m1 "${CONDA_PREFIX}/include/cudnn_version.h"
# define CUDNN_MAJOR 8
# define CUDNN_MINOR 4
# define CUDNN_PATCHLEVEL 1

python -c 'import tensorflow as tf; print(tf.config.list_physical_devices())'
# [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU')]

python -c 'import jax; print(jax.devices())'
# [StreamExecutorGpuDevice(id=0, process_index=0), ...]
```

## Cleanup
```bash
rm -r downloads imagenette
find . -type d -name __pycache__ -exec rm -r {} +
find . -type d -name .ipynb_checkpoints -exec rm -r {} +
```
