# Certification of Autoencoder-based Models for Dynamical Systems 

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/github/license/mledda17/AE-Certification-Method.svg)

Official code for the paper: **"Certification of Autoencoder-based Models for Dynamical Systems "**  

Authors:
- Marco Ledda, Student, IEEE
- Diego Deplano, Member, IEEE
- Alessandro Giua, Fellow, IEEE
- Mauro Franceschelli, Senior, IEEE

All authors are with the DIEE, University of Cagliari, 09123 Cagliari, Italy.

## 📝 Abstract

> Deep learning models have emerged as powerful tools for modeling complex dynamical systems, offering data-driven alternatives to traditional identification techniques.
Among them, autoencoder-based architectures have gained popularity due to their ability to extract low-dimensional latent representations starting from high-dimensional information. However, a major challenge persists: assessing the reliability of these models, especially in control tasks where prediction errors can have critical consequences. In this work, we propose an optimization-based certification approach to quantify the worst-case prediction error of ReLU-activated autoencoder models used for the identification of dynamical systems. By formulating a targeted Mixed-Integer Quadratic Programming, our approach identifies data sequences that maximize the deviation between the model's predicted output and the true system response.


## 🚀 Getting Started

### Setup
1. Clone the repository
2. Install requirements and dependecies
3. Get an API Key for Gurobi
4. Run experiments

### 1. Clone the repository
First of all, you will need to clone this repository. To do this, run the following command:

```bash
git clone https://github.com/mledda17/AE-Certification-Method.git
```

### 2. Install requirements and dependecies
You will need the following libraries to run the experiments:
- Python 3.10+
- Gurobipy 12.0.1+
- Matplotlib 3.10.1+
- Numpy 2.1.3+
- Tensorflow 2.19.0+
- Keras 2.13.1
- Scipy 1.15.2

Install dependencies via:

```bash
pip install -r requirements.txt
```

### 3. Get an API Key for Gurobi
You can get an API for Gurobi from the official website. Then, add in your .env file the key in the given format:
```bash
WLSACCESSID = [...]
WLSSECRET   = [...]
LICENSEID   = [...]
```

### 4. Run experiments
All experiments can be executed using the following command:
```bash
python3 main.py
```

The results of the experiments will be put in a new folder "certification_results/".

**🎉 Conference Acceptance**

> This paper has been accepted for presentation at the IEEE Conference on Decision and Control (CDC) 2025, to be held in Rio de Janeiro, Brazil, December 9–12, 2025.

Official code for the paper: **"Certification of Autoencoder-based Models for Dynamical Systems "**  
