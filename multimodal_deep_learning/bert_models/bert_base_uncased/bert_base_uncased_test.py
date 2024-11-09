import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm
import numpy as np

from dataset import (DATASET_BASE,
                     INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)
from utils.load_data import load_data_informative
from utils import device
import torchvision


def load_model(model_path, dict_path, device):
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.load_state_dict(torch.load(dict_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    return model


def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader):
            input_ids = d['text_input_ids'].to(device)
            attention_mask = d['text_attention_mask'].to(device)
            labels = d['text_labels'].to(device).long()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print((np.array(all_preds) == np.array(all_labels)).sum())
    return accuracy


if __name__ == "__main__":
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from bert_models.bert_base_uncased import BERT_BASE_UNCASED_PRETRAINED_PATH, BERT_BASE_UNCASED_FTMODEL_PATH

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH)

    _, _, test_loader = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=8
    )

    model_path = os.path.join(BERT_BASE_UNCASED_FTMODEL_PATH, 'bert_base_uncased_model.bin')
    model = load_model(BERT_BASE_UNCASED_PRETRAINED_PATH, model_path, device)

    accuracy = evaluate(model, test_loader, device)
    print(f'Validation Accuracy: {accuracy:.4f}')
