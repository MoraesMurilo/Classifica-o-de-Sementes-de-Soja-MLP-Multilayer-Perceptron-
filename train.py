import torch
import torch.nn as nn
import torch.optim as optim
from dataset_loader import load_datasets
from model import CNN

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Época [{epoch + 1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Treino Acc: {train_acc:.2f}%, Validação Acc: {val_acc:.2f}%")

    print("\nTreinamento finalizado!")
    torch.save(model.state_dict(), "modelo.pth")
    print("Modelo salvo como modelo.pth")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = load_datasets()
    model = CNN(num_classes=len(classes))
    train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=0.001, device=device)
