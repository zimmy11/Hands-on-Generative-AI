# 💡 Latent Diffusion Model (LDM) Light Implementation

## Project Overview

This repository hosts a from-scratch implementation of a light-version **Latent Diffusion Model (LDM)**, the core architecture behind state-of-the-art text-to-image synthesis models like Stable Diffusion. The project is structured across three phases, moving from basic unconditional training to a fully conditional text-to-image generator using **Classifier-Free Guidance (CFG)**.

[cite_start]The primary focus is on implementing the $\mathbf{U-Net}$ denoiser architecture from scratch, while leveraging pre-trained, lightweight components for the **Pixel Space (VAE)** and **Conditioning (Text Encoder)** to manage the project scope[cite: 1, 36, 43].

---

## 🏗️ Project Structure and File Descriptions

The project adheres to professional ML engineering standards, prioritizing **modularity**, **reproducibility**, and **experiment tracking** via **PyTorch Lightning** and **Weights & Biases (W&B)**.

### Top-Level Files and Directories

| File/Directory | Description |
| :--- | :--- |
| **`train.py`** | **Main Execution Script.** Handles argument parsing, loads configurations from `experiments/`, initializes the PyTorch Lightning `Trainer`, and starts the training process for the LDM. |
| **`requirements.txt`** | Pins all Python dependencies (e.g., `torch`, `pytorch-lightning`, `wandb`, `omegaconf`, `sentence-transformers`). |
| **`experiments/`** | Contains `.yaml` configuration files that define hyperparameters for specific training runs (e.g., learning rates, batch sizes, U-Net parameters, CFG drop rate). **Key to reproducibility.** |
| **`data/`** | Storage for dataset files. |
| **`models/`** | Storage for all model checkpoints, including the pre-trained VAE and the resulting trained U-Net $\epsilon_{\theta}$. |
| **`src/`** | **Source Code.** Contains all the core, clean, and reusable Python modules. |

### 📂 `src/` (Source Code Details)

The `src/` directory is the heart of the project, defining the model architectures and training logic.

#### `src/models/` (Model Architectures)

| File | Function | Phase |
| :--- | :--- | :--- |
| **`unet_model.py`** | [cite_start]Implements the $\mathbf{Denoising~U-Net}$ **from scratch**[cite: 33]. It assembles the core components (`ResBlocks`, `AttentionBlocks`, `Down/Upsample`) and includes stubs for cross-attention. | 1, 2, 3 |
| **`components.py`** | [cite_start]Defines fundamental building blocks like the $\mathbf{Residual~Block}$, the $\mathbf{Downsampling/Upsampling}$ modules, and the crucial $\mathbf{Cross-Attention}$ layer ($\mathbf{Q}, \mathbf{KV}$) for conditioning injection[cite: 1, 34, 44]. | 1, 3 |
| **`latent_diffusion.py`** | The main model wrapper, implemented as a $\mathbf{LightningModule}$. [cite_start]It contains the logic for the **forward diffusion process** and the **reverse (sampling) process** logic (e.g., $\text{DDPM}$ style sampling)[cite: 39, 40]. | 2, 3 |

#### `src/data/` (Data Pipelines)

| File | Function | Phase |
| :--- | :--- | :--- |
| **`vae_dataset.py`** | [cite_start]Handles loading raw images, applying necessary transformations, and encoding them into the latent space ($\mathbf{z}$) using the pre-trained VAE encoder ($\mathcal{E}$)[cite: 33]. | 1, 2 |
| **`conditional_dataset.py`** | Manages data for conditional training. [cite_start]Loads image-text pairs, tokenizes the text, and processes it for the lightweight pre-trained Text Encoder[cite: 43, 44]. | 3 |

#### `src/training/` (Training & Utilities)

| File | Function | Phase |
| :--- | :--- | :--- |
| **`trainer.py`** | [cite_start]Encapsulates the training loop logic, including the $\mathbf{DDPM~Loss~Objective}$, logging to $\text{W\&B}$, and managing the implementation of **Classifier-Free Guidance (CFG)** ($\text{randomly drop out the conditioning}$)[cite: 40, 44, 45]. | 2, 3 |
| **`utils/noise_scheduler.py`**| [cite_start]Implements the **fixed forward process** $\text{noise schedule}$ ($\beta_t$ values), essential for adding controlled Gaussian noise in the forward pass[cite: 40]. | 2 |

---

## 🛠 Phased Implementation Plan

The project is divided into three distinct phases over approximately 4 months, allowing for focused development and testing.

| Phase | Time | Focus | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **Phase 1: Architecture Setup** | [cite_start]3-4 weeks [cite: 32] | Set up the latent space. [cite_start]Implement the lightweight $\mathbf{U-Net}$ core components (downsampling, bottleneck, upsampling, skip connections) from scratch[cite: 33]. | A runnable $\text{U-Net}$ assembled from modular components. |
| **Phase 2: Unconditional LDM** | [cite_start]2-3 weeks [cite: 38] | Implement the **Forward** and **Reverse** diffusion processes. [cite_start]**Unconditionally train** the $\text{U-Net}$ on latent representations $\mathbf{z}$[cite: 39, 40]. | Model capable of generating images from pure noise $\mathbf{z}_T$ (latent space only). |
| **Phase 3: Conditional Text-to-Image** | [cite_start]3-4 weeks [cite: 42] | [cite_start]Integrate the pre-trained $\mathbf{Text~Encoder}$ (e.g., $\text{all-MiniLM-L6-v2}$) and modify the $\text{U-Net}$ to inject conditioning $\mathbf{c}$ via $\mathbf{Cross-Attention}$[cite: 43, 44]. [cite_start]Implement $\mathbf{Classifier-Free~Guidance (CFG)}$[cite: 44]. | Fully functional Text-to-Image LDM. |