# NavIA 🚢🤖

<img width="3548" height="1774" alt="image" src="https://github.com/user-attachments/assets/a95c5950-88fb-4928-a0d7-14b1729397f5" />

> **Bataille Navale & Intelligence Artificielle (Deep Learning avec PyTorch)**  
> Comparez et testez deux architectures d'apprentissage profond pour jouer à la bataille navale : un **MLP dense (V1)** et un **FCN convolutif (V2)**.

---

## 📌 Présentation du projet

**NavIA** est un projet d'expérimentation et de formation autour du Deep Learning appliqué au jeu classique de la bataille navale. L'objectif est d'entraîner des réseaux de neurones capables de prédire la case la plus probable où se trouve un navire adverse à partir de l'état connu de la grille, et d'observer en temps réel leurs stratégies de ciblage via des **cartes de chaleur (heatmaps)**.

Le projet propose deux générations de modèles :
- **NavIA V1 (MLP - Multi-Layer Perceptron)** : approche tabulaire avec un vecteur aplati de 100 cases.
- **NavIA V2 (FCN - Fully Convolutional Network)** : approche spatiale 2D avec tenseur à 3 canaux et couches de convolutions pour capturer la contiguïté et la géométrie des navires.

---

## 🧠 Comparaison des architectures

| Caractéristique | NavIA V1 (MLP) | NavIA V2 (FCN) |
|---|---|---|
| **Type de réseau** | Réseau dense entièrement connecté (Linear + ReLU) | Réseau entièrement convolutif (Conv2D + ReLU) |
| **Entrée** | Vecteur 1D de 100 cases (-1 = manqué, 0 = inexploré, 1 = touché) | Tenseur 3D `(3, 10, 10)` :<br>• Canal 0 : Touché (Hits)<br>• Canal 1 : Manqué (Misses)<br>• Canal 2 : Non exploré |
| **Structure spatiale** | Aplatit la grille (perte des relations de voisinage 2D) | Préserve la topologie 2D de la grille 10x10 |
| **Sortie** | 100 logits avec Sigmoid | Grille 10x10 de probabilités de présence |
| **Poids du modèle** | [`models/mlp_bataille_navale.pth`](models/mlp_bataille_navale.pth) (~290 Ko) | [`models/Navia_V2_FCN.pt`](models/Navia_V2_FCN.pt) (~155 Ko) |
| **Comportement** | Découverte heuristique globale | Traque efficace des cases adjacentes dès qu'un navire est touché |

---

## 🚀 Fonctionnalités

- **Interface Web Interactive** : Affrontez l'IA dans votre navigateur avec sélection dynamique du modèle (V1 MLP ou V2 FCN).
- **Heatmap dynamique en temps réel** : Visualisez sous forme de carte thermique la probabilité assignée par l'IA à chacune des cases de votre plateau.
- **API FastAPI REST** : Backend modulaire permettant de démarrer une partie, tirer et récupérer l'état du jeu ainsi que la matrice de heatmap.
- **Jeux en console (CLI)** : Scripts en ligne de commande pour jouer ou tester les deux modèles.
- **Générateurs de vidéos de démonstration** : Scripts de rendu OpenCV/Pillow simulant des duels complets IA vs IA ou Joueur vs IA (formats 16:9, 1:1 et 9:16 mobile).

---

## 📂 Structure du projet

```text
NavIA/
├── models/
│   ├── mlp_bataille_navale.pth    # Poids entraînés du modèle V1 (MLP)
│   └── Navia_V2_FCN.pt            # Poids entraînés du modèle V2 (FCN)
├── backend.py                     # API FastAPI multi-modèles (V1 & V2)
├── index.html                     # Interface web avec heatmap et sélecteur de modèle
├── train.py                       # Entraînement du modèle V1 (MLP) + logique de grille
├── eval.py                        # Partie CLI contre le modèle V1 (MLP)
├── Navia_V2_FCN_train.py          # Entraînement du modèle V2 (FCN)
├── use_FCN_Navia_V2.py            # Partie CLI contre le modèle V2 (FCN)
├── generate_vertical_demo.py      # Générateur de vidéo duel 9:16 (format mobile)
├── generate_demo.py               # Générateur de vidéo démo 16:9
├── generate_linkedin_demo.py      # Générateur de vidéo démo 1:1
├── requirements.txt               # Dépendances Python
└── README.md                      # Documentation du projet
```

---

## 🛠️ Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/jr534/NavIA.git
   cd NavIA
   ```

2. Installez les dépendances requises :
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Utilisation

### 1. Lancer l'interface Web (Recommandé)

Lancez le serveur backend FastAPI :
```bash
python backend.py
```
Puis ouvrez votre navigateur à l'adresse :
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

Vous pouvez choisir dans le menu déroulant :
- **Navia V2 (Convolutional FCN)** : le nouveau modèle convolutif.
- **Navia V1 (Dense MLP)** : le modèle historique dense.

### 2. Jouer dans le terminal (CLI)

- **Contre NavIA V2 (FCN)** :
  ```bash
  python use_FCN_Navia_V2.py
  ```

- **Contre NavIA V1 (MLP)** :
  ```bash
  python eval.py
  ```

---

## 🏋️ Entraînement des modèles

### Entraîner NavIA V2 (FCN)
Le script simule des parties contre une grille aléatoire et entraîne le réseau de neurones à chaque tir via rétropropagation :
```bash
python Navia_V2_FCN_train.py
```
Le meilleur modèle est automatiquement sauvegardé dans `models/Navia_V2_FCN.pt`.

### Entraîner NavIA V1 (MLP)
```bash
python train.py
```
Les poids sont sauvegardés dans `models/mlp_bataille_navale.pth`.

---

## 🎬 Visualisation & Vidéos de démonstration

<img width="800" height="450" alt="ezgif-3981abbd79d9e2c4" src="https://github.com/user-attachments/assets/0f8057d4-b773-4f7d-98ac-f60d02a58d98" />

Le projet contient des générateurs autonomes pour créer des vidéos de simulations de parties :
- **Format Vertical (9:16 - Mobile / Reels / TikTok)** :
  ```bash
  python generate_vertical_demo.py
  ```
- **Format Carré (1:1 - LinkedIn / Instagram)** :
  ```bash
  python generate_linkedin_demo.py
  ```
- **Format Paysage (16:9 - Standard)** :
  ```bash
  python generate_demo.py
  ```

---

## 🤖 Note sur l'utilisation de l'IA

Dans ce projet, les fichiers `backend.py` et `index.html` ont été codés avec l'aide d'une IA.

Cette utilisation s'inscrit en continuité avec l'objectif principal du projet, qui n'est pas de fournir une interface ou une API, mais de concevoir des réseaux de neurones (MLP et FCN) et de s'entraîner à leur création et leur évaluation.

---

## 📜 Licence

Projet open-source réalisé dans un objectif d'apprentissage et de formation autour du Deep Learning et de la vision par ordinateur.
