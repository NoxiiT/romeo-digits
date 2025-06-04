# Liste des datasets et le module de DataLoader associé
dATA_LOADERS = {
    "MNIST": {
        "module": "data_utils",
        "function": "get_mnist_dataloaders"
    },
    "Fashion-MNIST": {
        "module": "data_utils",
        "function": "get_fashionmnist_dataloaders"
    },
    "CIFAR-10": {
        "module": "data_utils",
        "function": "get_cifar10_dataloaders"
    }
}

# Modules modèles (création et prédiction) à importer dynamiquement
MODELS = {
    # Clés = nom affiché dans l'interface
    "SimpleCNN": {
        "module": "models.simple_cnn",
        "class": "SimpleCNN"
    },
    "CIFARCNN": {
        "module": "models.cifar_cnn",
        "class": "CIFARCNN"
    },
    # Pré-entraînés
    "ResNet50": {
        "module": "models.imagenet",
        "function": "classify_resnet50"
    },
    "YOLOv5": {
        "module": "models.yolo_models",
        "function": "classify_yolov5"
    },
    "YOLOv11": {
        "module": "models.yolo_models",
        "function": "classify_yolov11"
    }
}
