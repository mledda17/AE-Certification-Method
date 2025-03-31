# Certification of Autoencoder-based Models for Dynamical Systems 

Official code for the paper: **"Certification of Autoencoder-based Models for Dynamical Systems "**  

Authors:
- Marco Ledda, Student, IEEE
- Diego Deplano, Member, IEEE,
- Alessandro Giua, Fellow, IEEE
- Mauro Franceschelli, Senior, IEEE

## 📝 Abstract

> Deep learning models have emerged as powerful tools for modeling complex dynamical systems, offering data-driven alternatives to traditional identification techniques.
Among them, autoencoder-based architectures have gained popularity due to their ability to extract low-dimensional latent representations starting from high-dimensional information. However, a major challenge persists: assessing the reliability of these models, especially in control tasks where prediction errors can have critical consequences. In this work, we propose an optimization-based certification approach to quantify the worst-case prediction error of ReLU-activated autoencoder models used for the identification of dynamical systems. By formulating a targeted Mixed-Integer Quadratic Programming, our approach identifies data sequences that maximize the deviation between the model's predicted output and the true system response.


## 🚀 Getting Started

### Setup
1. Clone the repository
2. Install requirements and dependecies
3. Run experiments

### Clone the repository
First of all, you will need to clone this repository. To do this, run the following command:

```bash
git clone https://github.com/mledda17/AE-Certification-Method.git
```

### Install requirements and dependecies
You will need the following libraries to run the experiments:
- Python 3.10+
- Gurobipy 12.0.1+
- Matplotlib 3.10.1+
- Numpy 2.1.3+
- Tensorflow 2.19.0+
- Keras 2.9.0+
- Scipy 1.15.2

Install dependencies via:

```bash
pip install -r requirements.txt
```

### Run experiments
All experiments can be executed using the following command:
```bash
python3 main.py
```

