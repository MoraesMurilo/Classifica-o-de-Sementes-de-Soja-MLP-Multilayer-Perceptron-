import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Caminho do dataset conforme informado
data_dir = '/home/muri/Documents/PI/Dataset'

# Transformações de pré-processamento
# Data Augmentation + Normalização (valores típicos entre 0 e 1)
transform = transforms.Compose([
    transforms.Resize((64, 64)),            # Redimensiona as imagens para 64x64
    transforms.RandomHorizontalFlip(),      # Data Augmentation: flip horizontal aleatório
    transforms.RandomRotation(15),          # Data Augmentation: rotação aleatória de até 15 graus
    transforms.ToTensor(),                  # Converte imagem em tensor PyTorch
    transforms.Normalize([0.5], [0.5])      # Normaliza para [-1, 1] (escala comum para CNNs)
])

def load_datasets(batch_size=32, val_split=0.2, test_split=0.1):
    """Carrega os datasets de treino, validação e teste."""
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = dataset.classes

    return train_loader, val_loader, test_loader, class_names