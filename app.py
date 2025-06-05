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

# Méthode pour charger DataLoader dynamiquement
def load_dataloaders(dataset_name, batch_size, train_pct=0.8, val_pct=0.1):
    cfg = dATA_LOADERS[dataset_name]
    module = importlib.import_module(cfg["module"])
    fn = getattr(module, cfg["function"])
    return fn(batch_size, train_pct, val_pct)

# Méthode pour instancier ou appeler le modèle dynamiquement
def train_model(dataset_name, model_name, epochs, lr, batch_size, train_pct, val_pct):
    global MODEL, CLASSES, CURRENT_DATASET
    CURRENT_DATASET = dataset_name

    train_loader, val_loader, test_loader, num_classes = load_dataloaders(dataset_name, batch_size, train_pct, val_pct)
    cfg = MODELS[model_name]
    module = importlib.import_module(cfg["module"])
    if "class" in cfg:
        ModelClass = getattr(module, cfg["class"])
        MODEL = ModelClass(num_classes=num_classes)

        losses = []
        accs = []

        if model_name == "SimpleCNN":
            from models.simple_cnn import train_simplecnn
            train_fn = train_simplecnn
            train_args = dict(stream=True)
        else:
            from models.cifar_cnn import train_cifarcnn
            train_fn = train_cifarcnn
            train_args = dict(stream=True)

        for epoch, (loss, acc) in enumerate(train_fn(
            MODEL, train_loader, lr, epochs, DEVICE, **train_args
        )):
            losses.append(loss)
            accs.append(acc)
            df = pd.DataFrame([
                {"epoch": i + 1, "loss": losses[i], "accuracy": accs[i]}
                for i in range(len(losses))
            ])
            # Barre de progression HTML avec le style pip en monospace
            bar_len = 20
            done = int(bar_len * (epoch + 1) / epochs)
            remaining = bar_len - done - 1  # on réserve un caractère pour le marqueur "╸"

            filled = "━" * done
            marker = "╸"
            rest   = "━" * remaining  if remaining > 0 else ""
            
            colored = f'<span style="color:#f97316; font-family:monospace;">{filled}{marker}</span>'
            uncolored = f'<span style="color:#777; font-family:monospace;">{rest}</span>'
            bar_html = f'<span style="font-family:monospace;">{colored}{uncolored} {epoch+1}/{epochs}</span>'
            result_str = (
                f"{bar_html}<br>"
                f"Entraînement en cours : epoch {epoch+1}/{epochs}<br>"
                f"(train_pct={train_pct*100:.0f}%, val_pct={val_pct*100:.0f}%)<br>"
                f"Perte : {loss:.4f} | Précision : {acc*100:.2f}%<br>"
                f"Nombre de classes : {num_classes} | Modèle : {model_name} | Device : {DEVICE}"
            )
            #  ━━━━━━━━━━━━━━━━━━━━━━━╸━━━━━━━━━━━━━━━━
            yield result_str, df, df

        CLASSES = (
            [str(i) for i in range(num_classes)]
            if dataset_name != "CIFAR-10"
            else [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        )

        result_str = (
            f"Entraînement terminé sur {dataset_name} avec {epochs} epochs<br>"
            f"(train_pct={train_pct*100:.0f}%, val_pct={val_pct*100:.0f}%)<br>"
            f"Perte finale : {losses[-1]:.4f} | Précision finale : {accs[-1]*100:.2f}%<br>"
            f"Nombre de classes : {num_classes} | Modèle : {model_name} | Device : {DEVICE}"
        )
        df = pd.DataFrame([
            {"epoch": i + 1, "loss": losses[i], "accuracy": accs[i]}
            for i in range(len(losses))
        ])
        yield result_str, df, df

    else:
        yield "Impossible d'entraîner un modèle pré-entraîné.", None, None

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

def stream_yolov11(video):
    """
    Prend un flux vidéo (mp4) depuis la webcam, applique YOLOv11 sur chaque frame,
    et yield la vidéo annotée (mp4) en streaming.
    """
    from config import MODELS
    cfg = MODELS["YOLOv11"]
    module = importlib.import_module(cfg["module"])
    fn = getattr(module, cfg["function"])

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

def update_model_dropdown(selected_dataset):
    """
    Si l'utilisateur choisit CIFAR-10, on ne propose que CIFARCNN.
    Sinon on ne propose que SimpleCNN.
    """
    if selected_dataset == "CIFAR-10":
        return gr.update(choices=["CIFARCNN"], value="CIFARCNN")
    else:
        return gr.update(choices=["SimpleCNN"], value="SimpleCNN")

def yolov11_on_frame(frame, conf=0.25, iou=0.45):
    """
    Applique YOLOv11 sur une image PIL (frame webcam).
    """
    cfg = MODELS["YOLOv11"]
    module = importlib.import_module(cfg["module"])
    fn = getattr(module, cfg["function"])
    return fn(frame, conf, iou)

# --- Construction de l'interface Gradio --- #
with gr.Blocks() as demo:
    gr.Markdown("# ROMEO-DIGITS : Reconnaissance d’images & d’objets")

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
            epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
            lr = gr.Slider(0.0001, 0.01, value=0.001, step=0.0001, label="Learning Rate")
            batch_size = gr.Slider(8, 128, value=32, step=8, label="Batch Size")

        # Sliders pour proportion train/val/test
        with gr.Row():
            train_pct = gr.Slider(0.1, 0.9, value=0.8, step=0.05, label="Proportion train (%)", interactive=True)
            val_pct   = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Proportion validation (%)", interactive=True)

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
            inputs=[dataset, model, epochs, lr, batch_size, train_pct, val_pct],
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
                visible=False
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
        webcam_stream.stream(
            lambda frame, conf, iou: yolov11_on_frame(frame, conf, iou),
            inputs=[webcam_stream, conf_slider, iou_slider],
            outputs=output_img,
            time_limit=15,
            stream_every=0.1,
            concurrency_limit=30
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
