# Classificação de Sementes de Soja com CNN

Este projeto implementa uma Rede Neural Convolucional (CNN) em **PyTorch** para classificação automática de sementes de soja em diferentes categorias de qualidade. O objetivo é demonstrar a aplicação prática de **Deep Learning** em visão computacional, eliminando a necessidade de extração manual de características.

## Objetivo
Classificar sementes de soja em cinco classes distintas:
- Intact Soybeans (sementes intactas)
- Broken Soybeans (sementes quebradas)
- Immature Soybeans (sementes imaturas)
- Skin-Damaged Soybeans (sementes com danos superficiais)
- Spotted Soybeans (sementes com manchas)

## Dataset
- Origem: Kaggle - Soybean Seed Dataset
- Total: mais de 5.000 imagens reais de sementes
- Divisão: 70% treino, 20% validação, 10% teste

Estrutura esperada do diretório de dados:

Dataset/
├── Intact Soybeans/
├── Broken Soybeans/
├── Immature Soybeans/
├── Skin-damaged Soybeans/
└── Spotted Soybeans/


## Requisitos
Instale as dependências com:
```bash
pip install torch torchvision scikit-learn numpy

Estrutura do Projeto

PI/
├── Dataset/              # Subpastas com imagens organizadas por classe
├── modelo.pth            # Arquivo salvo do modelo treinado (gerado após o treino)
├── dataset_loader.py     # Código de pré-processamento e carregamento de dados
├── model.py              # Definição da arquitetura da CNN
├── train.py              # Script de treinamento
├── test.py               # Script de teste e avaliação
└── README.md

Execução
Treinar o modelo

python3 train.py

    Carrega e pré-processa as imagens

    Divide em treino/validação/teste

    Treina a CNN por 20 épocas

    Salva o modelo treinado em modelo.pth

Avaliar o modelo

python3 test.py

    Carrega o modelo salvo

    Avalia no conjunto de teste

    Exibe Acurácia, F1-Score e Matriz de Confusão

Resultados

Após 20 épocas de treinamento:

    Acurácia no teste: 84.57%

    F1-Score: 84.40%

As principais dificuldades observadas foram na diferenciação entre classes semelhantes, como sementes intactas vs. manchadas, e quebradas vs. imaturas.
Conclusão

O projeto demonstrou que uma CNN relativamente simples, combinada com pré-processamento e data augmentation, pode alcançar resultados sólidos na classificação automática de sementes de soja. O uso de Deep Learning se mostrou vantajoso ao aprender diretamente os padrões visuais sem necessidade de engenharia manual de atributos.
