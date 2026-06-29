# NavIA

NavIA est un projet de formation autour des reseaux de neurones, avec une application concrete : apprendre a un modele MLP (Multi-Layer Perceptron) a proposer des tirs dans une partie de bataille navale.

Le but du projet est pedagogique. Il sert a comprendre comment representer une grille de jeu, entrainer un reseau de neurones simple, exploiter ses predictions, puis relier ce modele a une interface jouable.

## Objectifs du projet

- Comprendre les bases d'un MLP applique a un probleme de recommandation.
- Representer une grille de bataille navale sous forme de donnees numeriques.
- Entrainer un modele PyTorch a estimer les cases les plus interessantes a cibler.
- Utiliser le resultat du modele pour produire une recommandation de tir plus personnalisee selon l'etat de la partie.
- Relier un backend Python a une interface web simple.

## Fonctionnement general

Le projet genere des grilles de bataille navale, entraine un modele MLP sur des parties simulees, puis utilise ce modele pour choisir les tirs de l'IA.

Pendant une partie, l'etat connu de la grille est transforme en vecteur de 100 valeurs. Le MLP renvoie ensuite un score pour chaque case. Les cases deja jouees sont ignorees, et la case avec le meilleur score devient la recommandation de tir de l'IA.

## Fichiers principaux

- [`train.py`](train.py) : definit la generation des grilles, la classe du modele MLP et la boucle d'entrainement.
- [`eval.py`](eval.py) : permet de tester le modele dans une partie en console.
- [`backend.py`](backend.py) : expose une API FastAPI pour lancer une partie, jouer un tir et obtenir la recommandation de l'IA.
- [`index.html`](index.html) : fournit l'interface web pour jouer contre l'IA.
- [`models/mlp_bataille_navale.pth`](models/mlp_bataille_navale.pth) : poids du modele entraine.

## Note sur l'utilisation de l'IA

Dans ce projet, les fichiers [`backend.py`](backend.py) et [`index.html`](index.html) ont ete entierement codes avec l'aide d'une IA.

Cette utilisation s'inscrit en continuité avec l'objectif principal du projet, qui n'est pas de fournir une interface ou une API, mais de concevoir un réseau de neurones MLP et de s'entraîner à sa création. 

## Technologies utilisees

- Python
- PyTorch
- FastAPI
- HTML, CSS et JavaScript
- Uvicorn

## Installation

Installez les dependances Python necessaires :

```bash
pip install -r requirements.txt
```

## Entrainer le modele

```bash
python train.py
```

Le script sauvegarde les poids du modele dans :

```text
models/mlp_bataille_navale.pth
```

## Lancer l'application web

```bash
python backend.py
```

Puis ouvrez l'application dans votre navigateur :

```text
http://127.0.0.1:8000
```

## Structure du projet

```text
NavIA/
+-- backend.py
+-- eval.py
+-- index.html
+-- train.py
+-- requirements.txt
+-- models/
|   +-- mlp_bataille_navale.pth
+-- README.md
```

## Pistes d'amelioration

- Corriger et uniformiser les noms de variables.
- Ajouter un fichier `requirements.txt`.
- Ajouter des tests unitaires sur la generation des grilles.
- Ameliorer l'entrainement avec plus de parties et une strategie de score plus riche.
- Afficher une heatmap des recommandations de l'IA dans l'interface.

## Licence

Projet realise dans un objectif de formation.
