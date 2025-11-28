[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-brightgreen.svg)](https://fastapi.tiangolo.com/)
[![Tensorflow](https://img.shields.io/badge/Tensorflow-2.x-orange.svg)](https://www.tensorflow.org)

# Beat Monitor – Neural Audio Fingerprinting (Version adaptée)

## À propos du projet
Beat Monitor est un système de **surveillance automatique des diffusions audio** destiné aux contextes où la collecte manuelle est insuffisante (radios, TV, maquis, lieux publics…).

Il utilise un modèle de **Neural Audio Fingerprinting (NeuralFP)** basé sur l’apprentissage contrastif pour :

- Générer des empreintes audio robustes
- Identifier automatiquement une musique à partir d’un court extrait
- Détecter les diffusions en temps réel ou en mode batch
- Aider à la rémunération équitable des artistes (cas d’usage BBDA)

Cette version est adaptée pour le **déploiement national au Burkina Faso**.

---

## ⚙️ Technologies utilisées

### **Backend**
- Python 3.9+
- FastAPI
- Uvicorn
- TensorFlow 2.x
- Libros
- angualar

### **Matching & Database**
- Faiss (Facebook AI)
- NumPy / SciPy
- Embeddings audio (NeuralFP)
- PostgresSQL

