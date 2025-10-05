# Model Capacity: Learning and Memorization

## Abstract

Deep networks can realize complex decision functions capable of memorizing the training set of standard image classification benchmarks, while under standard training regimes, they achieve state-of-the-art performance. The exact mechanism governing learning vs memorization in deep learning is currently not fully understood, and deep networks have shown in practice surprising phenomena that set them apart from traditional learning algorithms.

In this practical, we take a closer look at model capacity and memorization in deep learning, by reproducing established empirical phenomena of deep learning.

1. [Part I](./PartI.ipynb)
2. [Part II](./PartII.ipynb)

An optional task is described at the end of Part II.

## Setup

The assignment is based on the [Jax library](https://github.com/google/jax) for all tensor computations, as well as [Pytorch](https://pytorch.org/) for data loading. 

To configure the Python environment required for solving the assignment, you will need the following packages:
```
numpy
jax
flax
optax
torch
torchvision
matplotlib
jupyter-lab
seaborn
scipy
```

### GPU support
To speed up computations using GPUs, your system should provide the NVIDIA CUDA development runtime, consisting of:
```
NVIDIA GPU drivers
CUDA toolkit
CUDNN
```
Jax will rely on your system's CUDA installation to work. Importantly, unlike Tensorflow and Pytorch, JAX does not ship with a prepackaged cuda toolkit, which should thus be already installed on your system. Furthermore, Pytorch should be installed in the `cpuonly` version, since the version that bundles `cudatoolkit` would conflict with your CUDA system installation, causing Jax to fail to detect your GPU.

### Anaconda

If you have Anaconda (or Miniconda) installed, you can install all requirements by using the provided `environment.yml` file.
```bash
conda env create -f environment.yml
```
This will create the environment `dd2412-memorization`.

Finally, you can install jax by running:
```bash
conda activate dd2412-memorization
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Please check the [official Jax installation guide](https://github.com/google/jax#pip-installation-gpu-cuda) for choosing the right version corresponding to your CUDA installation.

### Pip

If using `pip`, you can install all the assignment dependencies by using the provded `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Finally, you can install Jax by running:
```bash
pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
Please check the [official Jax installation guide](https://github.com/google/jax#pip-installation-gpu-cuda) for choosing the right version corresponding to your CUDA installation.

## Testing your installation
If all was setup correctly, Jax should be able to see your GPU. You can test it by running:
```bash
conda activate dd2412-memorization # if using conda
python -c 'import jax; import torch; print(jax.devices()); print(torch.cuda.is_available())'
```
which should output:
```
[GpuDevice(id=0, process_index=0)]
True
```

## Troubleshooting
If your Jax installation fails to detect your GPU, it means that either:
* The incorrect version of Pytorch was installed (the version of the `torch` package, showing with `pip freeze` should not report `cuda`.
* The cuda toolkit is not in your path. Refer to the [Jax installation](https://github.com/google/jax#pip-installation-gpu-cuda) instructions for details.
