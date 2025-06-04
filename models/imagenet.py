import torch
from torchvision import models, transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_IMAGENET = models.resnet50(pretrained=True).to(DEVICE)
MODEL_IMAGENET.eval()

imagenet_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# Charger les labels ImageNet depuis un fichier local
with open("imagenet_classes.txt", "r") as f:
    IMAGENET_LABELS = [line.strip() for line in f.readlines()]


def classify_resnet50(image):
    if image is None:
        return "Aucune image fournie."
    img_tensor = imagenet_transforms(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL_IMAGENET(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
    top5_prob, top5_catid = torch.topk(probs, 5)
    results = []
    for i in range(top5_prob.size(0)):
        label = IMAGENET_LABELS[top5_catid[i]]
        score = top5_prob[i].item()
        results.append(f"{label} : {score*100:.1f}%")
    return "\n".join(results)
