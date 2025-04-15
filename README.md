# 🧠 Energy-Based GAN (EBGAN) on CIFAR-10  
*By Achraf Guenounou*

## 📘 Overview

This project implements an **Energy-Based Generative Adversarial Network (EBGAN)** to generate realistic images from the CIFAR-10 dataset. Unlike traditional GANs, EBGAN uses an **autoencoder-based discriminator** that assigns an energy score to each sample based on reconstruction error, promoting a more stable training process.

---

## 🔍 What is EBGAN?

EBGAN is a GAN variant where:
- The **generator** produces fake images aiming to minimize their energy score.
- The **discriminator** is an **autoencoder** that reconstructs real/fake images and assigns an energy value based on reconstruction quality.

### ✳️ Pulling-away Term (PT)
A regularization technique that avoids mode collapse by penalizing similar feature vectors in the generated batch using cosine similarity.

---

## 🧱 Model Architecture

### 🔷 Generator
- Dense → Reshape to (8×8×128)
- Conv2DTranspose → (16×16×128) → BatchNorm + ReLU
- Conv2DTranspose → (32×32×64) → BatchNorm + ReLU
- Conv2DTranspose → (32×32×3) → Tanh

### 🔷 Discriminator
- Image input + Class input → Concatenated
- Conv2D → (16×16×128) → LeakyReLU
- Conv2D → (8×8×128) → BatchNorm + LeakyReLU
- Conv2D → (4×4×64) → BatchNorm + LeakyReLU
- Flatten → Dense(64) → Dense(1)

---

## 🧪 Training Data

**Dataset**: CIFAR-10  
- 60,000 color images (32×32 pixels, 10 classes)  
- **Training set**: 50,000 images  
- **Testing set**: 10,000 images  
- **Preprocessing**: Normalization to [-1, 1] and reshaping

---

## ⚙️ Model Parameters

| **Parameter**            | **Value**                 |
|--------------------------|--------------------------|
| Optimization Algorithm   | Adam                     |
| Learning Rate            | 0.0001                   |
| Beta_1                   | 0.5                      |
| Epochs                   | 200                      |
| Batch Size               | 256                      |
| Noise Vector Size        | 100                      |
| Discriminator Loss       | Energy-Based Loss        |
| Generator Loss           | Minimize Energy Score    |

---

## 📈 Performance

- **EBGAN** with normalized energy loss produced more stable and visually realistic outputs than baseline CGAN using binary cross-entropy.
- Demonstrated **more stable training curves**.
- Generated images improved across epochs.

---

## 🖼️ Sample Outputs

Images were generated at regular intervals during training.  
Results show **higher visual quality with normalization** and the use of the **energy function**.

---

## ✅ Summary

- Minimizes energy scores of generated samples  
- More stable training than traditional GANs  
- Suited for semi-supervised learning  

---

## 🔧 Future Improvements

- Implement a fully functional autoencoder structure  
- Test with more complex datasets (e.g., CelebA, STL-10)  
- Apply to new domains like **anomaly detection** or **network intrusion detection**  

---

## 📚 References

- [Energy-Based GANs (Zhao et al., 2016)](https://arxiv.org/pdf/1609.03126)  
- [Sik-Ho Tsang’s Review on EBGAN](https://sh-tsang.medium.com/review-ebgan-energy-based-generative-adversarial-network-gan-9dcc4bba9d7b)

---

## 🙋‍♂️ Author

**Achraf Guenounou**  
Neural Network Lab — February 2025
