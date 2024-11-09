from fusion_models.bert_resnet50 import FUSION_MODELS_SAVE_DIR
from utils import device
from utils.load_data import load_data_informative
from fusion_models.bert_resnet50.fusion_model import FusionModel
from bert_models.bert_encoder import BertEncoder
from cnn_models.resnet50.cnn_resnet_train import ResNet50Encoder

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from transformers import BertTokenizer
from tqdm import tqdm
import os


def evaluate(model, data_loader, criterion, bert_encoder, cnn_encoder):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            texts, texts_attention_mask, images, image_conf_norm, labels = (batch['text_input_ids'].to(device),
                                                                            batch['text_attention_mask'].to(device),
                                                                            batch['image'].to(device),
                                                                            batch['image_info_conf_norm'].to(device),
                                                                            batch['text_labels'].to(device))
            text_vector = bert_encoder.encode(texts, texts_attention_mask)
            image_vector = cnn_encoder.encode(images)
            # inputs = torch.cat((text_vector, image_vector, image_conf.unsqueeze(1)), dim=1)

            outputs = model(text_vector, image_vector, image_conf_norm.unsqueeze(1))
            loss = criterion(outputs, labels.unsqueeze(1))
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            truth = (labels > 0.5).float()
            correct_predictions += (predicted == truth.unsqueeze(1)).sum().item()
            total_predictions += truth.size(0)

    avg_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy


def train(model, train_loader, dev_loader, criterion, optimizer, num_epochs, bert_encoder, cnn_encoder):
    model.to(device)
    best_acc = 0.0
    best_model_path = os.path.join(FUSION_MODELS_SAVE_DIR, 'best_fusion_model.pth')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader):
            texts, texts_attention_mask, images, image_conf_norm, labels = (batch['text_input_ids'].to(device),
                                                                            batch['text_attention_mask'].to(device),
                                                                            batch['image'].to(device),
                                                                            batch['image_info_conf_norm'].to(device),
                                                                            batch['text_labels'].to(device))

            text_vector = bert_encoder.encode(texts, texts_attention_mask)
            image_vector = cnn_encoder.encode(images)

            optimizer.zero_grad()

            outputs = model(text_vector, image_vector, image_conf_norm.unsqueeze(1))

            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if torch.isnan(loss):
                print(batch['image_path'])
                raise Exception

        avg_train_loss = running_loss / len(train_loader)

        avg_dev_loss, acc = evaluate(model, dev_loader, criterion, bert_encoder, cnn_encoder)
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Dev Loss: {avg_dev_loss:.4f}, Dev Accuracy: {acc:.4f}')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

    print('Finished Training')
    print(f'Best model saved to {best_model_path} with accuracy: {best_acc:.4f}')


# ----------------------------------------------------------------------------------
continue_train = True
model_path = os.path.join(FUSION_MODELS_SAVE_DIR, 'best_fusion_model.pth')

if __name__ == '__main__':
    num_epochs = 10
    model = FusionModel()
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    # criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from bert_models.bert_base_uncased import (BERT_BASE_UNCASED_PRETRAINED_PATH,
                                               BERT_BASE_UNCASED_FTMODEL_PATH)

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH)

    bert_encoder = BertEncoder(model_path=BERT_BASE_UNCASED_PRETRAINED_PATH,
                               ft_model_path=BERT_BASE_UNCASED_FTMODEL_PATH,
                               device=device)

    from cnn_models.resnet50 import RESNET50_SAVED_MODELS_BASE

    cnn_encoder = ResNet50Encoder().to(device)
    state_dict = torch.load(os.path.join(RESNET50_SAVED_MODELS_BASE, 'resnet50_model_1.pth'),
                            map_location='cpu',
                            weights_only=True)
    cnn_encoder.load_state_dict(state_dict)
    print("Loaded ResNet50 weight")

    from dataset import (INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                         INFORMATIVE_DATASET_DEV_MULTI_PATH,
                         INFORMATIVE_DATASET_TEST_MULTI_PATH)

    train_loader, dev_loader, test_loader = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=256
    )

    if continue_train:
        state_dict = torch.load(model_path,
                                map_location='cpu',
                                weights_only=True)
        model.load_state_dict(state_dict)

    train(model=model,
          train_loader=train_loader,
          dev_loader=dev_loader,
          criterion=criterion,
          optimizer=optimizer,
          num_epochs=num_epochs,
          bert_encoder=bert_encoder,
          cnn_encoder=cnn_encoder)
