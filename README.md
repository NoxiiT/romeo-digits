# ROMEO-DIGITS

Application interactive pour la reconnaissance d’images et d’objets avec différents modèles de deep learning (PyTorch, Gradio).

## Fonctionnalités

- Entraînement de modèles CNN sur MNIST, Fashion-MNIST, CIFAR-10
- Test de modèles pré-entraînés (ResNet50, YOLOv5, YOLOv11)
- Visualisation des réseaux de neurones
- Interface utilisateur simple avec Gradio
- Prédiction d’images via upload ou webcam

## Installation

1. Clonez ce dépôt :
   ```bash
   git clone <repo-url>
   cd romeo_digits
   ```
2. Installez les dépendances :
   ```bash
   # graphviz est nécessaire pour visualiser les modèles
   sudo apt-get install graphviz
   pip install -r requirements.txt
   ```

## Lancement

```bash
python app.py
```

L’interface Gradio s’ouvrira dans votre navigateur.

## Structure du projet

- `app.py` : application principale Gradio
- `models/` : modèles et fonctions de classification
- `data_utils.py` : chargement des datasets
- `config.py` : configuration des modèles et datasets

```
romeo_digits/
├── app.py
├── config.py
├── data_utils.py
├── models/
│   ├── __init__.py
│   ├── simple_cnn.py
│   ├── cifar_cnn.py
│   ├── imagenet.py
│   └── yolo_models.py
└── requirements.txt
```

---
Projet CHPS1005
