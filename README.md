[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-brightgreen.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)

# Beat Monitor – Neural Audio Fingerprinting (Version adaptée)

## À propos du projet

### Le défi
Comment garantir une **rémunération équitable des artistes** quand la surveillance manuelle des diffusions est impossible à l'échelle nationale ?

### Notre réponse
**Beat Monitor**, développé avec **DataLab**, déploie l'IA au service de la propriété intellectuelle. Le système **NeuralFP** identifie automatiquement chaque œuvre diffusée, qu'elle passe sur une radio en ligne ou une télévision.

### Le gain
✅ **Zéro déclaration manuelle**  
✅ **Traçabilité totale**  
✅ **Droits d'auteur calculés automatiquement** et redistribués équitablement  

**Pour le BBDA (Bureau Burkinabé du Droit d'Auteur), c'est une révolution.**

---

## 🎯 Fonctionnalités clés

Beat Monitor utilise un modèle de **Neural Audio Fingerprinting (NeuralFP)** basé sur l'apprentissage contrastif pour :

- 🎵 Générer des empreintes audio robustes
- 🔍 Identifier automatiquement une musique à partir d'un court extrait
- ⏱️ Détecter les diffusions en temps réel ou en mode batch
- 💰 Aider à la rémunération équitable des artistes

Cette version est adaptée pour le **déploiement national au Burkina Faso**.

---

## 🎵 Modèle source

Ce projet utilise le code source du modèle **Neural Audio Fingerprinting** développé par :

**📦 Dépôt officiel :** [github.com/mimbres/neural-audio-fp](https://github.com/mimbres/neural-audio-fp)

### Modifications principales apportées

| Aspect | Version originale | Notre adaptation |
|--------|------------------|------------------|
| **Stratégie d'identification** | Requêtes ponctuelles (query-based) | **Surveillance continue en streaming** |
| **Fonction de loss** | Cross-Entropy (Xent) | **Triplet Loss** (meilleure robustesse) |
| **Cas d'usage** | Identification à la demande | Monitoring 24/7 multi-radios |

### Pourquoi ces changements ?

✅ **Triplet Loss** : Améliore la séparation des embeddings et la robustesse au bruit  
✅ **Surveillance continue** : Détection automatique sans intervention manuelle  
✅ **Architecture temps réel** : Streaming audio + détection par segments consécutifs

---

## ⚙️ Technologies utilisées

### **Backend**
- Python 3.9+
- FastAPI (API REST)
- Uvicorn (serveur ASGI)
- Tensorflow 2.x (modèle NeuralFP)
- Librosa (traitement audio)
- FFmpeg (décodage streams)

### **Frontend**
- Angular 18+
- PrimeNG (UI components)
- Chart.js (visualisations)
- WebSocket (temps réel)

### **Matching & Database**
- Faiss (Facebook AI Similarity Search)
- SQLite (stockage détections)
- NumPy / SciPy (calculs scientifiques)
- Embeddings audio 128D (NeuralFP)