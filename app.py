import gradio as gr
import importlib
import pandas as pd
from config import dATA_LOADERS, MODELS
from PIL import Image
import torch
import tempfile
import cv2
import numpy as np
import warnings
import logging
import os

# Suppress NNPACK warnings from PyTorch
warnings.filterwarnings("ignore", message="Could not initialize NNPACK")
logging.getLogger("torch").setLevel(logging.ERROR)

# Détection du device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Variables globales
MODEL = None
CLASSES = []
CURRENT_DATASET = None

empty_df = pd.DataFrame({
    "epoch":   pd.Series(dtype="int"),
    "loss":    pd.Series(dtype="float"),
    "accuracy": pd.Series(dtype="float")
})

# Dossier pour sauvegarder les modèles
SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def get_model_save_path(dataset_name, model_name):
    # Nom de fichier unique selon dataset et modèle
    return os.path.join(SAVED_MODELS_DIR, f"{dataset_name}_{model_name}.pt")

def save_model(model, dataset_name, model_name):
    path = get_model_save_path(dataset_name, model_name)
    torch.save(model.state_dict(), path)

def load_model(model, dataset_name, model_name):
    path = get_model_save_path(dataset_name, model_name)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        return True
    return False

def list_saved_models():
    # Retourne la liste des modèles sauvegardés
    if not os.path.exists(SAVED_MODELS_DIR):
        return []
    return [
        f.replace(".pt", "")
        for f in os.listdir(SAVED_MODELS_DIR)
        if f.endswith(".pt")
    ]

# Méthode pour charger DataLoader dynamiquement
def load_dataloaders(dataset_name, batch_size, train_pct=0.8, val_pct=0.1, num_workers=0):
    cfg = dATA_LOADERS[dataset_name]
    module = importlib.import_module(cfg["module"])
    fn = getattr(module, cfg["function"])
    return fn(batch_size, train_pct, val_pct, num_workers=num_workers)

# Méthode pour instancier ou appeler le modèle dynamiquement
def train_model(
    dataset_name, model_name, epochs, lr, batch_size, train_pct, val_pct, num_workers,
    num_conv, conv1_filters, conv2_filters, conv3_filters, conv4_filters, conv5_filters,
    num_dense, dense1_units, dense2_units, dense3_units,
    load_existing_model=False, selected_saved_model=None
):
    global MODEL, CLASSES, CURRENT_DATASET
    CURRENT_DATASET = dataset_name

    # Sécuriser les valeurs None
    num_conv = num_conv if num_conv is not None else 1
    num_dense = num_dense if num_dense is not None else 1
    conv1_filters = conv1_filters if conv1_filters is not None else 32
    conv2_filters = conv2_filters if conv2_filters is not None else 64
    conv3_filters = conv3_filters if conv3_filters is not None else 128
    conv4_filters = conv4_filters if conv4_filters is not None else 128
    conv5_filters = conv5_filters if conv5_filters is not None else 128
    dense1_units = dense1_units if dense1_units is not None else 128
    dense2_units = dense2_units if dense2_units is not None else 64
    dense3_units = dense3_units if dense3_units is not None else 32

    # 1️⃣ Charger les DataLoaders
    train_loader, val_loader, test_loader, num_classes = load_dataloaders(
        dataset_name, batch_size, train_pct, val_pct, num_workers
    )

    # 2️⃣ Charger la classe du modèle
    cfg = MODELS[model_name]
    module = importlib.import_module(cfg["module"])
    ModelClass = getattr(module, cfg["class"])

    # 3️⃣ Construire dynamiquement conv_layers
    conv_filters = []
    if num_conv >= 1: conv_filters.append(conv1_filters)
    if num_conv >= 2: conv_filters.append(conv2_filters)
    if num_conv >= 3: conv_filters.append(conv3_filters)
    if num_conv >= 4: conv_filters.append(conv4_filters)
    if num_conv >= 5: conv_filters.append(conv5_filters)
    conv_layers = [{"out_channels": f, "kernel_size": 3} for f in conv_filters]

    # 4️⃣ Construire dynamiquement dense_layers
    dense_units = []
    if num_dense >= 1: dense_units.append(dense1_units)
    if num_dense >= 2: dense_units.append(dense2_units)
    if num_dense >= 3: dense_units.append(dense3_units)
    dense_layers = dense_units

    # 5️⃣ Instancier le modèle avec la config
    MODEL = ModelClass(
        num_classes=num_classes,
        conv_layers=conv_layers,
        dense_layers=dense_layers
    )

    # Charger un modèle sauvegardé si demandé
    if load_existing_model and selected_saved_model:
        # Extraire dataset/model depuis le nom sauvegardé
        try:
            ds, mdl = selected_saved_model.split("_", 1)
            load_model(MODEL, ds, mdl)
        except Exception:
            pass

    CLASSES = (
        [str(i) for i in range(num_classes)]
        if dataset_name != "CIFAR-10"
        else [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]
    )

    # 6️⃣ Choisir la fonction d'entraînement
    if model_name == "SimpleCNN":
        from models.simple_cnn import train_simplecnn as train_fn
    else:
        from models.cifar_cnn import train_cifarcnn as train_fn

    losses = []
    accs = []

    # 7️⃣ Boucle streaming
    for epoch, (loss, acc) in enumerate(train_fn(
        MODEL, train_loader, lr, epochs, DEVICE, stream=True
    )):
        losses.append(loss)
        accs.append(acc)

        # DataFrame progressif
        df = pd.DataFrame([
            {"epoch": i+1, "loss": losses[i], "accuracy": accs[i]}
            for i in range(len(losses))
        ])

        # Barre pip-style en monospace + couleur
        bar_len   = 20
        done      = int(bar_len * (epoch+1)/epochs)
        remaining = bar_len - done - 1
        filled    = "━"*done
        marker    = "╸"
        rest      = "━"*remaining if remaining>0 else ""
        colored   = f'<span style="color:#f97316; font-family:monospace;">{filled}{marker}</span>'
        uncolored = f'<span style="color:#777; font-family:monospace;">{rest}</span>'
        bar_html  = f'<span style="font-family:monospace;">[{colored}{uncolored}] {epoch+1}/{epochs}</span>'
        
        device_str = f"<span style='color:#22c55e;'>GPU</span>" if DEVICE.type != "cpu" else f"<span style='color:#f43f5e;'>CPU</span>"
        
        result_str = (
            f"{bar_html}<br>"
            f"Appareil utilisé : {device_str}<br>"
            f"Epoch {epoch+1}/{epochs} — "
            f"(train {train_pct*100:.0f}%, val {val_pct*100:.0f}%)<br>"
            f"Loss : {loss:.4f} | Acc : {acc*100:.2f}%"
        )
        yield result_str, df, df

    # 8️⃣ Message final + courbe complète
    device_str = f"<span style='color:#22c55e;'>GPU</span>" if DEVICE.type != "cpu" else f"<span style='color:#f43f5e;'>CPU</span>"
    final_str = (
        f"✔️ Entraînement terminé sur {dataset_name} en {epochs} epochs<br>"
        f"Appareil utilisé : {device_str}<br>"
        f"Loss finale : {losses[-1]:.4f} | Acc finale : {accs[-1]*100:.2f}%"
    )
    df = pd.DataFrame([
        {"epoch": i+1, "loss": losses[i], "accuracy": accs[i]}
        for i in range(len(losses))
    ])
    # Sauvegarder le modèle entraîné
    save_model(MODEL, dataset_name, model_name)
    yield final_str + "<br>Modèle sauvegardé.", df, df

def predict(image):
    global MODEL, CLASSES, CURRENT_DATASET
    if MODEL is None:
        return "Erreur : aucun modèle n'a été entraîné."

    # Prétraitements pour MNIST / CIFAR
    import torchvision.transforms as transforms
    if CURRENT_DATASET in ["MNIST", "Fashion-MNIST"]:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    MODEL.eval()
    with torch.no_grad():
        outputs = MODEL(img_tensor)
        _, predicted = outputs.max(1)
        label = CLASSES[predicted.item()]

    return f"Classe prédite : {label}"

# Chargement dynamique des fonctions de classification pour pré-entraînés
def classify_pretrained(model_name, image, conf=0.25, iou=0.45):
    cfg = MODELS[model_name]
    module = importlib.import_module(cfg["module"])
    fn = getattr(module, cfg["function"])
    # YOLOv11 a besoin de seuils supplémentaires
    if model_name == "YOLOv11":
        return None, fn(image, conf, iou)
    else:
        return fn(image), None

# Ajout d'une variable globale pour le modèle YOLOv11
YOLOV11_MODEL_FN = None

def get_yolov11_model_fn():
    global YOLOV11_MODEL_FN
    if YOLOV11_MODEL_FN is None:
        cfg = MODELS["YOLOv11"]
        module = importlib.import_module(cfg["module"])
        YOLOV11_MODEL_FN = getattr(module, cfg["function"])
    return YOLOV11_MODEL_FN

def stream_yolov11(video):
    """
    Prend un flux vidéo (mp4) depuis la webcam, applique YOLOv11 sur chaque frame,
    et yield la vidéo annotée (mp4) en streaming.
    """
    # Utilise le modèle déjà chargé
    fn = get_yolov11_model_fn()

    # Ouvre la vidéo d'entrée
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Temp file pour la vidéo annotée
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
        out = cv2.VideoWriter(tmpfile.name, fourcc, fps, (width, height))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convertir BGR->RGB puis PIL.Image
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Appliquer YOLOv11 (doit retourner une image annotée PIL.Image)
            annotated = fn(img, 0.25, 0.45)
            # Convertir PIL.Image -> BGR np.array
            annotated_np = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
            out.write(annotated_np)
            frames.append(annotated_np)
            # Pour le streaming, yield la vidéo temporaire toutes les N frames
            if len(frames) % int(fps) == 0:
                out.release()
                yield tmpfile.name
                out = cv2.VideoWriter(tmpfile.name, fourcc, fps, (width, height))
        out.release()
        yield tmpfile.name
    cap.release()

def yolov11_on_frame(frame, conf=0.25, iou=0.45):
    """
    Applique YOLOv11 sur une image PIL (frame webcam).
    """
    fn = get_yolov11_model_fn()
    # Redimensionner ici AVANT l'inférence pour accélérer le pipeline
    frame = frame.resize((640, 384))
    return fn(frame, conf, iou)

def update_model_dropdown(selected_dataset):
    """
    Si l'utilisateur choisit CIFAR-10, on ne propose que CIFARCNN.
    Sinon on ne propose que SimpleCNN.
    """
    if selected_dataset == "CIFAR-10":
        return gr.update(choices=["CIFARCNN"], value="CIFARCNN")
    else:
        return gr.update(choices=["SimpleCNN"], value="SimpleCNN")

def plot_model(
    model_name, num_conv, conv1_filters, conv2_filters, conv3_filters, conv4_filters, conv5_filters,
    num_dense, dense1_units, dense2_units, dense3_units
):
    import tempfile
    from torchviz import make_dot
    import PIL.Image

    # Sécuriser les valeurs None
    num_conv = num_conv if num_conv is not None else 1
    num_dense = num_dense if num_dense is not None else 1
    conv1_filters = conv1_filters if conv1_filters is not None else 32
    conv2_filters = conv2_filters if conv2_filters is not None else 64
    conv3_filters = conv3_filters if conv3_filters is not None else 128
    conv4_filters = conv4_filters if conv4_filters is not None else 128
    conv5_filters = conv5_filters if conv5_filters is not None else 128
    dense1_units = dense1_units if dense1_units is not None else 128
    dense2_units = dense2_units if dense2_units is not None else 64
    dense3_units = dense3_units if dense3_units is not None else 32

    # Construire la config du modèle
    conv_filters = []
    if num_conv >= 1: conv_filters.append(conv1_filters)
    if num_conv >= 2: conv_filters.append(conv2_filters)
    if num_conv >= 3: conv_filters.append(conv3_filters)
    if num_conv >= 4: conv_filters.append(conv4_filters)
    if num_conv >= 5: conv_filters.append(conv5_filters)
    conv_layers = [{"out_channels": f, "kernel_size": 3} for f in conv_filters]

    dense_units = []
    if num_dense >= 1: dense_units.append(dense1_units)
    if num_dense >= 2: dense_units.append(dense2_units)
    if num_dense >= 3: dense_units.append(dense3_units)
    dense_layers = dense_units

    # Instancier le modèle
    cfg = MODELS[model_name]
    module = importlib.import_module(cfg["module"])
    ModelClass = getattr(module, cfg["class"])
    model = ModelClass(num_classes=10, conv_layers=conv_layers, dense_layers=dense_layers)

    # Dummy input selon le modèle
    if model_name == "SimpleCNN":
        dummy = torch.zeros(1, 1, 28, 28)
    else:
        dummy = torch.zeros(1, 3, 32, 32)

    # Vérification de la taille de sortie des features
    try:
        with torch.no_grad():
            feat = model.features(dummy)
        if feat.shape[-1] == 0 or feat.shape[-2] == 0:
            raise ValueError("La configuration choisie réduit la taille de l'image à zéro. Diminuez le nombre de couches convolutionnelles ou la taille du pooling.")
    except Exception as e:
        # Créer une image d'erreur temporaire
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            img = np.ones((100, 400, 3), dtype=np.uint8) * 255
            import cv2
            cv2.putText(img, "Erreur: sortie trop petite!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imwrite(tmpfile.name, img)
            return tmpfile.name

    # Graphe torchviz
    model.eval()
    out = model(dummy)
    dot = make_dot(out, params=dict(list(model.named_parameters())))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        dot.format = "png"
        dot.render(tmpfile.name, cleanup=True)
        return tmpfile.name + ".png"

# --- Construction de l'interface Gradio --- #
with gr.Blocks() as demo:
    gr.Markdown("# ROMEO-DIGITS : Reconnaissance d'images & d'objets")

    # --- Onglet 1 : Entraîner son propre modèle --- #
    with gr.Tab("Créer ton propre modèle"):
        with gr.Row():
            dataset = gr.Dropdown(
                choices=["MNIST", "Fashion-MNIST", "CIFAR-10"],
                label="Choisir un dataset",
                value="MNIST"
            )
            model = gr.Dropdown(
                choices=["SimpleCNN"],
                label="Choisir un modèle",
                value="SimpleCNN"
            )
        dataset.change(update_model_dropdown, inputs=dataset, outputs=model)

        with gr.Row():
            epochs = gr.Slider(1, 100, value=5, step=1, label="Epochs")
            lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
            batch_size = gr.Slider(8, 8192, value=128, step=32, label="Batch Size")
            num_workers = gr.Slider(0, 8, value=0, step=1, label="Num Workers (DataLoader)")

        # Sliders pour proportion train/val/test
        with gr.Row():
            train_pct = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Proportion train (%)", interactive=True)
            val_pct   = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Proportion validation (%)", interactive=True)

        with gr.Row():
            num_conv       = gr.Slider(1, 5, value=2, step=1, label="Nombre de couches convolutionnelles")
            conv1_filters  = gr.Slider(8, 256, value=32, step=8, label="Filtres couche 1")
            conv2_filters  = gr.Slider(8, 256, value=64, step=8, label="Filtres couche 2")
            conv3_filters  = gr.Slider(8, 256, value=128, step=8, label="Filtres couche 3", visible=False)
            conv4_filters  = gr.Slider(8, 256, value=128, step=8, label="Filtres couche 4", visible=False)
            conv5_filters  = gr.Slider(8, 256, value=128, step=8, label="Filtres couche 5", visible=False)

        # Correction ici : afficher les sliders d'indice < num_conv
        def toggle_conv_sliders(n):
            return [
                gr.update(visible=(i < n))
                for i in range(5)
            ]
        num_conv.change(
            fn=toggle_conv_sliders,
            inputs=num_conv,
            outputs=[conv1_filters, conv2_filters, conv3_filters, conv4_filters, conv5_filters]
        )

        with gr.Row():
            num_dense      = gr.Slider(1, 3, value=1, step=1, label="Nombre de couches denses")
            dense1_units   = gr.Slider(16, 2048, value=128, step=16, label="Unités dense 1")
            dense2_units   = gr.Slider(16, 2048, value=64,  step=16, label="Unités dense 2", visible=False)
            dense3_units   = gr.Slider(16, 2048, value=32,  step=16, label="Unités dense 3", visible=False)

        def toggle_dense_sliders(n):
            return [
                gr.update(visible=(i < n))
                for i in range(3)
            ]
        num_dense.change(
            fn=toggle_dense_sliders,
            inputs=num_dense,
            outputs=[dense1_units, dense2_units, dense3_units]
        )

        # Ajout d'une option pour charger un modèle existant
        with gr.Row():
            load_existing_model = gr.Checkbox(label="Charger un modèle sauvegardé", value=False)
            saved_models_dropdown = gr.Dropdown(
                choices=list_saved_models(),
                label="Modèles sauvegardés",
                visible=False
            )

        def update_saved_models_visibility(checked):
            return gr.update(visible=checked)

        load_existing_model.change(
            fn=update_saved_models_visibility,
            inputs=load_existing_model,
            outputs=saved_models_dropdown
        )

        def refresh_saved_models():
            return gr.update(choices=list_saved_models())

        demo.load(
            fn=refresh_saved_models,
            inputs=None,
            outputs=saved_models_dropdown
        )

        train_output = gr.HTML(
            label="Résultat de l'entraînement",
            min_height=100,
        )
        with gr.Row():
            train_plot = gr.LinePlot(
                value=empty_df,
                x="epoch", y="loss",
                title="Courbes de perte",
                overlay=True, width=500, height=300
            )
            train_plot_acc = gr.LinePlot(
                value=empty_df,
                x="epoch", y="accuracy",
                title="Courbes de précision",
                overlay=True, width=500, height=300
            )
        train_btn = gr.Button("Lancer l'entraînement")
        train_btn.click(
            fn=train_model,
            inputs=[
                dataset, model, epochs, lr, batch_size, train_pct, val_pct, num_workers,
                num_conv, conv1_filters, conv2_filters, conv3_filters, conv4_filters, conv5_filters,
                num_dense, dense1_units, dense2_units, dense3_units,
                load_existing_model, saved_models_dropdown
            ],
            outputs=[train_output, train_plot, train_plot_acc]
        )

        gr.Markdown("---")
        gr.Markdown("## 🔍 Tester le modèle (MNIST / CIFAR-10)")
        with gr.Row():
            img_input = gr.Image(
                sources=["upload", "webcam", "clipboard"],
                type="pil",
                label="Uploader ou webcam"
            )
        predict_btn = gr.Button("Prédire")
        prediction = gr.HTML(label="Résultat de la prédiction")
        predict_btn.click(fn=predict, inputs=[img_input], outputs=prediction)

        gr.Markdown("---")
        gr.Markdown("## 📊 Visualiser le schéma du modèle")
        with gr.Row():
            model_image = gr.Image(label="Schéma du modèle", type="filepath")
        plot_btn = gr.Button("Afficher le schéma")
        plot_btn.click(
            fn=plot_model,
            inputs=[
                model, num_conv, conv1_filters, conv2_filters, conv3_filters, conv4_filters, conv5_filters,
                num_dense, dense1_units, dense2_units, dense3_units
            ],
            outputs=[model_image]
        )

    # --- Onglet 2 : Essayer des modèles pré-entraînés --- #
    with gr.Tab("Essayer des modèles pré-entraînés"):
        gr.Markdown("## 🔍 Essayer des modèles pré-entraînés")
        with gr.Row():
            existing_models = gr.Dropdown(
                choices=["ResNet50", "YOLOv5", "YOLOv11"],
                label="Choisir un modèle pré-entraîné",
                value="ResNet50"
            )
        model_header = gr.Markdown("### Modèle : ResNet50 (ImageNet)")
        existing_models.change(
            fn=lambda m: {
                "ResNet50": "### Modèle : ResNet50 (ImageNet)",
                "YOLOv5":   "### Modèle : YOLOv5 (détection)",
                "YOLOv11":  "### Modèle : YOLOv11 (détection)"
            }.get(m),
            inputs=existing_models,
            outputs=model_header
        )
        with gr.Row():
            img_pre = gr.Image(
                sources=["upload", "webcam", "clipboard"], 
                type="pil", 
                label="Image d'entrée",
                visible=True
            )
            webcam_stream = gr.Image(
                sources="webcam",
                type="pil",
                label="Webcam YOLOv11",
                visible=False,
                streaming=True  # Ajout : active le mode streaming pour la webcam
            )
            conf_slider = gr.Slider(
                0.01, 1.0, value=0.25, step=0.01,
                label="Confiance (YOLOv11)", visible=False
            )
            iou_slider = gr.Slider(
                0.01, 1.0, value=0.45, step=0.01,
                label="IoU (YOLOv11)", visible=False
            )
            output_img = gr.Image(
                label="Image annotée YOLOv11",
                visible=False
            )
        classify_btn   = gr.Button("Classifier")
        classif_output = gr.Textbox(
            label="Résultats texte",
            lines=5,
            interactive=False
        )

        def toggle_inputs(model):
            # Affiche la webcam et l'image annotée seulement pour YOLOv11
            return (
                gr.update(visible=(model != "YOLOv11")),  # img_pre
                gr.update(visible=(model == "YOLOv11")),  # webcam_stream
                gr.update(visible=(model == "YOLOv11")),  # conf_slider
                gr.update(visible=(model == "YOLOv11")),  # iou_slider
                gr.update(visible=(model == "YOLOv11")),  # output_img
                gr.update(visible=(model != "YOLOv11"))   # classif_output
            )

        existing_models.change(
            fn=toggle_inputs,
            inputs=existing_models,
            outputs=[img_pre, webcam_stream, conf_slider, iou_slider, output_img, classif_output]
        )

        # Pour YOLOv11 webcam : streaming direct frame->frame
        # Correction ici : on retourne l'image annotée à chaque frame
        webcam_stream.stream(
            fn=yolov11_on_frame,
            inputs=[webcam_stream, conf_slider, iou_slider],
            outputs=output_img,
            # Les paramètres time_limit, stream_every, concurrency_limit restent inchangés
            time_limit=None,               # <--- PAS de limite de temps, ou mets une grande valeur (ex: 60)
            stream_every=0.5,                # <--- Plus rapide (30 ms entre chaque frame, ~33 FPS max)
            concurrency_limit=2
        )

        # Pour les autres modèles ou image YOLOv11 (upload)
        def classify_pretrained_or_image(model, image, conf, iou):
            if model == "YOLOv11" and image is not None:
                _, img = classify_pretrained(model, image, conf, iou)
                return img, None
            elif model != "YOLOv11":
                txt, _ = classify_pretrained(model, image)
                return None, txt
            else:
                return None, None

        classify_btn.click(
            fn=classify_pretrained_or_image,
            inputs=[existing_models, img_pre, conf_slider, iou_slider],
            outputs=[output_img, classif_output]
        )


if __name__ == "__main__":
    demo.launch(share=True)
