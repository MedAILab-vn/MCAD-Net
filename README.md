# MCAD-Net: Multi-modal Cross-Attention Distillation Network 

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Accepted-brightgreen.svg)]() Official PyTorch implementation of the paper: **"A Multi-Scale Cross-Attention Network with
Knowledge Distillation for Retinal Disease Classification"**.

## 📖 Abstract
Diagnosing ophthalmic diseases often requires comprehensive analysis of both Color Fundus Photography (CFP) and Optical Coherence Tomography (OCT). We propose **MCAD-Net**, a novel Multi-modal Cross-Attention Distillation Network designed to effectively fuse robust representations from both modalities. By leveraging a bidirectional spatial cross-attention mechanism and handling shallow/deep semantic features, MCAD-Net consistently outperforms existing state-of-the-art single-modal and multi-modal baseline methods.

## ✨ Key Features
* **Bidirectional Cross-Attention:** Effectively fuses spatial features between CFP and OCT streams without collapsing spatial dimensions.
* **Feature Distillation & Selective Freezing:** Optimizes training by discarding task-agnostic low-level features and expanding deep semantic representations.
* **Uncertainty-Aware Learning (UAL):** Integrates Monte Carlo Dropout (MC-Dropout) for robust predictions and uncertainty estimation.
* **High Performance:** 

## 🏗️ Architecture


## ⚙️ Installation

**1. Clone the repository:**
```bash
git clone https://github.com/MedAILab-vn/MCAD-Net.git
cd MCAD-Net
