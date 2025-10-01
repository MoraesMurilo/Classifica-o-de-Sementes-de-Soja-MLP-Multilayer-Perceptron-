import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from dataset_loader import load_datasets
from model import CNN

def test_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Acurácia no Teste: {acc*100:.2f}%")
    print(f"F1-Score: {f1*100:.2f}%")
    print("Matriz de Confusão:")
    print(cm)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, classes = load_datasets()
    model = CNN(num_classes=len(classes))
    model.load_state_dict(torch.load("/home/muri/Documents/PI/modelo.pth"))
    print("Modelo carregado com sucesso!\n")
    test_model(model, test_loader, device=device)
