# DCGAN on Fashion-MNIST (PyTorch Implementation)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO_NAME/blob/main/DCGAN.ipynb)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

A PyTorch implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) trained on the Fashion-MNIST dataset.

This project is a reimplementation of the Keras-based DCGAN found in **Chapter 9 of "Deep Learning for Vision Systems"** by Mohamed Elgendy. It replicates the architecture and training logic of the original [Keras-GAN](https://github.com/eriklindernoren/Keras-GAN/tree/master/dcgan) repository but utilizes PyTorch's dynamic computational graph.

## 🖼️ Results
<img width="989" height="990" alt="download" src="https://github.com/user-attachments/assets/026fd531-d8b7-40c7-b1e4-b03170ec252f" />

## 🧠 Architecture

The architecture follows the standard DCGAN guidelines (Radford et al.):

### Generator
* **Input:** Latent vector $z$ (size 100).
* **Structure:** A series of `Upsample` and `Conv2d` layers with Batch Normalization and ReLU activation.
* **Output:** 28x28 Grayscale image (Values between -1 and 1 via `Tanh`).

### Discriminator
* **Input:** 28x28 Grayscale image.
* **Structure:** Strided `Conv2d` layers for downsampling, LeakyReLU activations, and Dropout for regularization.
* **Output:** Probability (Real vs. Fake) via `Sigmoid`.

## 🛠️ Implementation Details (Keras vs. PyTorch)

If you are coming from the book's Keras implementation, here are the key differences in this version:

1.  **Data Loading:** Uses `torch.utils.data.DataLoader` instead of manual NumPy slicing.
2.  **Training Loop:** Manual optimization steps (`optimizer.step()`) and gradient zeroing (`zero_grad()`) replace Keras' `model.train_on_batch()`.
3.  **Computational Graph:** Uses `.detach()` on generated images when training the Discriminator to prevent gradients from flowing back into the Generator.
4.  **Weights:** Weights are initialized (explicitly or implicitly) and updated using the Adam optimizer with `betas=(0.5, 0.999)` to prevent mode collapse.


## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch
* Torchvision
* Matplotlib
* Jupyter Notebook / Google Colab


### Installation
```bash
pip install torch torchvision matplotlib
```

### Usage
The easiest way to run this is via the provided Jupyter Notebook.
1. Clone the repository.
2. Open DCGAN.ipynb.
3. Run all cells.

The training loop will save images every save_interval epochs to visualize the Generator's progress.

📚 References
1. Book: Deep Learning for Vision Systems by Mohamed Elgendy.
2. Original Code (Keras): eriklindernoren/Keras-GAN
   
📝 License
This project is open-source and available for educational purposes.
