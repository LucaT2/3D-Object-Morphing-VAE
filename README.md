# 3D-Object-Morphing-VAE 🤖

A 3D Variational Autoencoder (VAE) trained on the **Modelnet10** dataset that can **morph and interpolate 3D objects** across 10 distinct categories. This project demonstrates the power of latent space manipulation for generating novel 3D geometry. This model used many training techniques that are in this paper: [Generative and Discriminative Voxel Modeling with Convolutional Neural Networks](https://arxiv.org/abs/1608.04236), most important being the modified loss function that penalizes the model if it generates wrong empty voxels.

## 🚀 Live Demo & Code

| Resource | Link | Description |
| :--- | :--- | :--- |
| **Web App Playground** | [https://lucat2-3d-vae-playground.hf.space/](https://lucat2-3d-vae-playground.hf.space/) | Interact with the trained model using a **Gradio** frontend. |
| **Kaggle Notebook** | [https://www.kaggle.com/code/tasadanluca/3d-object-morphing](https://www.kaggle.com/code/tasadanluca/3d-object-morphing) | View the complete development, training, and analysis code. |

## ✨ Features

* **3D Object Morphing:** Seamlessly transition between two different 3D objects by interpolating their latent space representations.
* **Variational Autoencoder (VAE):** Utilizes a VAE architecture for learning a smooth, continuous latent space, essential for meaningful interpolation.
* **Modelnet10 Dataset:** Trained on 10 common object categories (e.g., chair, table, sofa, monitor).

## 🛠️ Technology Stack

* **Deep Learning Backend:** **Python** and **TensorFlow**
* **Frontend/Interface:** **Gradio** (used for the interactive web app)
* **Data:** Modelnet10 ( Voxelized representations)

## ➡️ Getting Started

I recommend using an evironment, such as conda or if you prefer you can also you a local one. Also in order for the tensorflow library to work with your gpu, you either must
run this project from linux, or if you are on Windows you should run it from a Linux subsystem such as WSL or Ubuntu. Otherwise your training or evaluating phase will not be able to run on gpu.

To use, train, or evaluate the model locally, follow the instructions below. 

### 1. Setup

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/LucaT2/3D-Object-Morphing-VAE
cd 3D-Object-Morphing-VAE
pip install -r requirements.txt
```

### 2. Start the training process for the 3D VAE
```bash
python train_main.py
```
### 3. Evaluate the Model
```bash
python eval_main.py
```

### 4.Run the app locally 
(Though there should be no need for this as I already provided a link to the app above)
```bash
python app.py
```
