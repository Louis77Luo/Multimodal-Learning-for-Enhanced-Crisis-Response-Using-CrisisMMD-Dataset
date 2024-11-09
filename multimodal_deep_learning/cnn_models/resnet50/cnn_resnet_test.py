from cnn_models.resnet50 import RESNET50_SAVED_MODELS_BASE
from utils.load_data import load_data_informative
from dataset import (DATASET_BASE,
                     INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)
from utils import device

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torchsummary import summary

from sklearn.metrics import confusion_matrix

from transformers import BertTokenizer
from tqdm import tqdm
from PIL import Image
import pandas as pd
import os


class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet18 = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # Remove the fully connected layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])

        # Add a new fully connected layer for binary classification
        self.fc = nn.Linear(512, 1)

    def encode(self, x):
        # Forward pass through the ResNet18 model
        with torch.no_grad():
            features = self.resnet18(x)
            # Flatten the output tensor to a 1D vector
            features = features.view(features.size(0), -1)
            return features

    def forward(self, x):
        features = self.resnet18(x)
        features = features.view(features.size(0), -1)
        # Pass the features through the fully connected layer
        output = self.fc(features)
        output = torch.sigmoid(output)
        return output


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader):
        images, labels = (
            batch['image'].to(device),
            batch['image_label'].to(device),
        )
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        predicted = (outputs.squeeze() > 0.5).float()
        labels = (labels > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = running_loss / len(train_loader)
    return average_loss, accuracy


def validate_model(model, val_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, labels = (
                batch['image'].to(device),
                batch['image_label'].to(device),
            )
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
    average_loss = running_loss / len(val_loader)

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

    _, _, test_loader = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=64
    )

    model = ResNet18Encoder().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    state_dict = torch.load(os.path.join(RESNET50_SAVED_MODELS_BASE, 'resnet_regression_model_1.pth'),
                            map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    loss, acc = validate_model(model, test_loader, criterion, device)
    print(loss, acc)
