import torch
import torch.nn as nn
import torch.optim as optim

class CIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CIFARCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Fonction d'entraînement similaire à celle de simple_cnn
def train_cifarcnn(model, train_loader, lr, epochs, device, progress=None, on_epoch=None):
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
    return losses, accs
