from fusion_models.bert_resnet18 import FUSION_MODELS_SAVE_DIR
from utils import device
from utils.load_data import load_data_informative
from fusion_models.bert_resnet18.fusion_model import FusionModel
from bert_models.bert_encoder import BertEncoder
from cnn_models.resnet18.cnn_resnet_train import ResNet18Encoder

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from transformers import BertTokenizer
from tqdm import tqdm
import os

from sklearn.metrics import precision_score, recall_score, confusion_matrix, average_precision_score

def evaluate(model, data_loader, criterion, bert_encoder, cnn_encoder):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            texts, texts_attention_mask, images, image_conf_norm, labels = (batch['text_input_ids'].to(device),
                                                                            batch['text_attention_mask'].to(device),
                                                                            batch['image'].to(device),
                                                                            batch['image_info_conf_norm'].to(device),
                                                                            batch['text_labels'].to(device))
            text_vector = bert_encoder.encode(texts, texts_attention_mask)
            image_vector = cnn_encoder.encode(images)

            outputs = model(text_vector, image_vector, image_conf_norm.unsqueeze(1))
            loss = criterion(outputs, labels.unsqueeze(1))
            running_loss += loss.item()

            predicted = (outputs > 0.5).float()
            truth = (labels > 0.5).float()
            correct_predictions += (predicted == truth.unsqueeze(1)).sum().item()
            total_predictions += truth.size(0)

            all_labels.extend(truth.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    accuracy = correct_predictions / total_predictions

    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    mAP = average_precision_score(all_labels, all_predictions)

    return avg_loss, accuracy, precision, recall, mAP, conf_matrix


# ----------------------------------------------------------------------------------
model_path = os.path.join(FUSION_MODELS_SAVE_DIR, '8413.pth')

if __name__ == '__main__':
    model = FusionModel().to(device)
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)

    criterion = nn.BCELoss()

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

    from cnn_models.resnet18 import RESNET18_SAVED_MODELS_BASE

    cnn_encoder = ResNet18Encoder().to(device)
    state_dict = torch.load(os.path.join(RESNET18_SAVED_MODELS_BASE, 'resnet_regression_model_1.pth'), map_location='cpu', weights_only=True)
    cnn_encoder.load_state_dict(state_dict)
    print("Loaded ResNet18 weight")

    from dataset import (INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                         INFORMATIVE_DATASET_DEV_MULTI_PATH,
                         INFORMATIVE_DATASET_TEST_MULTI_PATH)

    _, dev_loader, _ = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=256
    )

    loss, acc, precision, recall, mAP, conf_matrix = evaluate(model=model,
                             data_loader=dev_loader,
                             criterion=criterion,
                             bert_encoder=bert_encoder,
                             cnn_encoder=cnn_encoder)

    print(loss, acc, precision, recall, mAP)
    print(conf_matrix)