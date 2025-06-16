import torch
from PIL import Image

print("[YOLO MODELS] Importing torch and PIL.Image")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[YOLO MODELS] Using device: {DEVICE}")

try:
    from ultralytics import YOLO
    YOLOV5_AVAILABLE = True
    print("[YOLO MODELS] ultralytics.YOLO imported successfully")
except ImportError:
    YOLOV5_AVAILABLE = False
    print("[YOLO MODELS] ultralytics not available, falling back to torch.hub")
    import torch.hub

# Chargement YOLOv5
try:
    if YOLOV5_AVAILABLE:
        print("[YOLO MODELS] Loading YOLOv5s.pt with ultralytics.YOLO")
        YOLOV5_MODEL = YOLO("yolov5s.pt")
        YOLOV5_MODEL.to(DEVICE)  # Move to GPU if available
    else:
        print("[YOLO MODELS] Loading YOLOv5s from torch.hub")
        YOLOV5_MODEL = torch.hub.load('ultralytics/yolov5:v6.2', 'yolov5s', pretrained=True, force_reload=True)
        YOLOV5_MODEL = YOLOV5_MODEL.to(DEVICE)  # Move to GPU if available
    YOLOV5_LOAD_ERROR = None
    print("[YOLO MODELS] YOLOv5 loaded successfully")
except Exception as e:
    YOLOV5_MODEL = None
    YOLOV5_LOAD_ERROR = str(e)
    print(f"[YOLO MODELS] Error loading YOLOv5: {YOLOV5_LOAD_ERROR}")

# Chargement YOLOv11
try:
    if YOLOV5_AVAILABLE:
        print("[YOLO MODELS] Loading YOLOv11n.pt with ultralytics.YOLO")
        YOLOV11_MODEL = YOLO("yolo11n.pt")
        YOLOV11_MODEL.to(DEVICE)  # Move to GPU if available
        YOLOV11_LOAD_ERROR = None
        print("[YOLO MODELS] YOLOv11 loaded successfully")
    else:
        YOLOV11_MODEL = None
        YOLOV11_LOAD_ERROR = "YOLOv11 n'est disponible qu'avec ultralytics >= 8.x et le fichier yolo11n.pt."
        print("[YOLO MODELS] YOLOv11 not available with torch.hub")
except Exception as e:
    YOLOV11_MODEL = None
    YOLOV11_LOAD_ERROR = str(e)
    print(f"[YOLO MODELS] Error loading YOLOv11: {YOLOV11_LOAD_ERROR}")


def classify_yolov5(image):
    print("[YOLO MODELS] classify_yolov5 called")
    if YOLOV5_LOAD_ERROR:
        print(f"[YOLO MODELS] YOLOV5_LOAD_ERROR: {YOLOV5_LOAD_ERROR}")
        return f"Erreur YOLOv5 : {YOLOV5_LOAD_ERROR}"
    if image is None:
        print("[YOLO MODELS] No image provided to classify_yolov5")
        return "Aucune image fournie."
    # Ensure inference is on the right device
    # if hasattr(YOLOV5_MODEL, "to"):
    #     print(f"[YOLO MODELS] Moving YOLOV5_MODEL to device: {DEVICE}")
    #     YOLOV5_MODEL.to(DEVICE)
    print("[YOLO MODELS] Running inference with YOLOv5")
    results = YOLOV5_MODEL(image)
    print(f"[YOLO MODELS] YOLOv5 inference results: {results}")
    boxes = results[0].boxes
    print(f"[YOLO MODELS] YOLOv5 boxes: {boxes}")
    if boxes is None or len(boxes) == 0:
        print("[YOLO MODELS] No objects detected by YOLOv5")
        return "Aucun objet détecté."
    lines = []
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = YOLOV5_MODEL.model.names[cls] if hasattr(YOLOV5_MODEL.model, "names") else str(cls)
        lines.append(f"{label} : {conf*100:.1f}%")
        print(f"[YOLO MODELS] Detected: {label} ({conf*100:.1f}%)")
    return "\n".join(lines)


def classify_yolov11(image, conf_threshold=0.25, iou_threshold=0.45):
    print("[YOLO MODELS] classify_yolov11 called")
    if YOLOV11_LOAD_ERROR:
        print(f"[YOLO MODELS] YOLOV11_LOAD_ERROR: {YOLOV11_LOAD_ERROR}")
        return YOLOV11_LOAD_ERROR
    if image is None:
        print("[YOLO MODELS] No image provided to classify_yolov11")
        return "Aucune image fournie."
    # Ensure inference is on the right device
    # if hasattr(YOLOV11_MODEL, "to"):
    #     print(f"[YOLO MODELS] Moving YOLOV11_MODEL to device: {DEVICE}")
    #     YOLOV11_MODEL.to(DEVICE)
    # Fix: Use 0 if DEVICE.index is None and CUDA is used
    device_arg = (
        DEVICE.index if DEVICE.type == "cuda" and DEVICE.index is not None
        else (0 if DEVICE.type == "cuda" else "cpu")
    )
    print(f"[YOLO MODELS] Using device: {DEVICE}, DEVICE.index: {DEVICE.index if DEVICE.type == 'cuda' else 'cpu'}")
    print(f"[YOLO MODELS] YOLOv11 predict params: conf={conf_threshold}, iou={iou_threshold}, imgsz=640, device='cuda'")
    results = YOLOV11_MODEL.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
        device='cuda'
    )
    print(f"[YOLO MODELS] YOLOv11 inference results: {results}")
    for r in results:
        print(f"[YOLO MODELS] Processing result: {r}")
        im_array = r.plot()
        print(f"[YOLO MODELS] Result image array shape: {im_array.shape}")
        im = Image.fromarray(im_array[..., ::-1])
        print("[YOLO MODELS] Returning annotated image from YOLOv11")
        return im
    print("[YOLO MODELS] Aucun résultat from YOLOv11")
    return "Aucun résultat."
