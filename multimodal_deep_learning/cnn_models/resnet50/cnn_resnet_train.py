from cnn_models.resnet50 import RESNET50_SAVED_MODELS_BASE
from utils.load_data import load_data_informative
from dataset import (DATASET_BASE,
                     INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)
from utils import device

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
from torchsummary import summary
from transformers import BertTokenizer
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from PIL import Image
import pandas as pd
import os


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        self.resnet50 = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)

    def encode(self, x):
        with torch.no_grad():
            features = self.resnet50(x)
            # Flatten the output tensor to a 1D vector
            features = features.view(features.size(0), -1)
            return features

    def forward(self, x):
        x = self.resnet50(x)
        x = x.view(x.size(0), -1)
        # Pass the features through the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def train_batch(model, batch, criterion, optimizer, device):
    model.train()
    images, labels = batch['image'].to(device), batch['image_label'].to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs.squeeze(), labels)
    loss.backward()
    optimizer.step()
    predicted = (outputs.squeeze() > 0.5).float()
    labels = (labels > 0.5).float()
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return loss.item(), correct, total


def train_model(model, train_loader, dev_loader, criterion, optimizer, device, num_epochs):
    best_accuracy = 0.0
    metrics = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader):
            loss, batch_correct, batch_total = train_batch(model, batch, criterion, optimizer, device)
            running_loss += loss
            correct += batch_correct
            total += batch_total

        train_accuracy = correct / total
        train_loss = running_loss / len(train_loader)

        dev_loss, dev_accuracy = validate_model(model, dev_loader, criterion, device)

        metrics.append(f"Epoch [{epoch + 1}/{num_epochs}]\n"
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}\n"
                       f"Val Loss  : {dev_loss:.4f}, Val Acc : {dev_accuracy:.4f}\n")

        print(metrics[-1])

        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            torch.save(model.state_dict(), os.path.join(RESNET50_SAVED_MODELS_BASE, 'best_model.pth'))
            print(f'Saved best model at acc={best_accuracy}')

    with open(os.path.join(RESNET50_SAVED_MODELS_BASE, 'training_metrics.txt'), 'w') as f:
        f.writelines(metrics)

    print("Training complete.")


def validate_model(model, dev_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dev_loader):
            images, labels = batch['image'].to(device), batch['image_label'].to(device)

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            labels = (labels > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = correct / total
    average_loss = running_loss / len(dev_loader)

    cm = confusion_matrix(all_labels, all_predictions)
    print(cm)

    return average_loss, accuracy


if __name__ == '__main__':
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from bert_models.bert_base_uncased import BERT_BASE_UNCASED_PRETRAINED_PATH

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH)

    train_loader, dev_loader, test_loader = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=128
    )

    model = ResNet50Encoder().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # torch.save(model.state_dict(), os.path.join(RESNET50_SAVED_MODELS_BASE, 'resnet50_model_0.pth'))
    # print("Model saved")
    # raise Exception

    num_epochs = 10

    continue_train = False
    if continue_train:
        state_dict = torch.load(os.path.join(RESNET50_SAVED_MODELS_BASE, 'resnet50_model_1.pth'), map_location='cpu',
                                weights_only=True)
        model.load_state_dict(state_dict)

    train_model(model, train_loader, dev_loader, criterion, optimizer, device, num_epochs)
