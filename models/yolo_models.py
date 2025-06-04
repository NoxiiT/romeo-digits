import torch
from PIL import Image

try:
    from ultralytics import YOLO
    YOLOV5_AVAILABLE = True
except ImportError:
    YOLOV5_AVAILABLE = False
    import torch.hub

# Chargement YOLOv5
try:
    if YOLOV5_AVAILABLE:
        YOLOV5_MODEL = YOLO("yolov5s.pt")
    else:
        YOLOV5_MODEL = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    YOLOV5_LOAD_ERROR = None
except Exception as e:
    YOLOV5_MODEL = None
    YOLOV5_LOAD_ERROR = str(e)

# Chargement YOLOv11
try:
    if YOLOV5_AVAILABLE:
        YOLOV11_MODEL = YOLO("yolo11n.pt")
        YOLOV11_LOAD_ERROR = None
    else:
        YOLOV11_MODEL = None
        YOLOV11_LOAD_ERROR = "YOLOv11 n'est disponible qu'avec ultralytics >= 8.x et le fichier yolo11n.pt."
except Exception as e:
    YOLOV11_MODEL = None
    YOLOV11_LOAD_ERROR = str(e)


def classify_yolov5(image):
    if YOLOV5_LOAD_ERROR:
        return f"Erreur YOLOv5 : {YOLOV5_LOAD_ERROR}"
    if image is None:
        return "Aucune image fournie."
    results = YOLOV5_MODEL(image)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return "Aucun objet détecté."
    lines = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = YOLOV5_MODEL.model.names[cls] if hasattr(YOLOV5_MODEL.model, "names") else str(cls)
        lines.append(f"{label} : {conf*100:.1f}%")
    return "\n".join(lines)


def classify_yolov11(image, conf_threshold=0.25, iou_threshold=0.45):
    if YOLOV11_LOAD_ERROR:
        return YOLOV11_LOAD_ERROR
    if image is None:
        return "Aucune image fournie."
    results = YOLOV11_MODEL.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        return im
    return "Aucun résultat."
