import torch
import torch.nn as nn
import torch.optim as optim

class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10, conv_layers=None, dense_layers=None):
        super(CIFARCNN, self).__init__()
        # Valeurs par défaut strictes demandées
        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel_size": 3},
                {"out_channels": 64, "kernel_size": 3},
                {"out_channels": 128, "kernel_size": 3}
            ]
        if dense_layers is None:
            dense_layers = [256]

        layers = []
        in_channels = 3
        for i, cfg in enumerate(conv_layers):
            layers.append(nn.Conv2d(in_channels, cfg["out_channels"], cfg.get("kernel_size", 3), padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = cfg["out_channels"]
        self.features = nn.Sequential(*layers)

        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            feat_shape = self.features(dummy).shape
        flat_size = feat_shape[1] * feat_shape[2] * feat_shape[3]

        # Valeur par défaut pour la première couche dense
        dense = [nn.Flatten()]
        in_features = flat_size
        for units in dense_layers:
            dense.append(nn.Linear(in_features, units))
            dense.append(nn.ReLU())
            in_features = units
        dense.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*dense)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Fonction d'entraînement similaire à celle de simple_cnn
def train_cifarcnn(model, train_loader, lr, epochs, device, progress=None, on_epoch=None, stream=False):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        losses.append(epoch_loss)
        accs.append(epoch_acc)
        if progress is not None:
            progress((epoch+1)/epochs, desc=f"Epoch {epoch+1}/{epochs}")
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss, epoch_acc)
        if stream:
            yield epoch_loss, epoch_acc
    if not stream:
        return losses, accs
