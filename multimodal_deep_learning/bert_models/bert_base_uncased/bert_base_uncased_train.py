from dataset import (INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)
from utils import device
from utils.load_data import load_data_informative


import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, average_precision_score
import numpy as np
from tqdm import tqdm
import re
import os
import argparse

def train_epoch(model, data_loader, optimizer, device, scheduler):
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for d in tqdm(data_loader):  # 使用 tqdm 包装 data_loader
        input_ids = d['text_input_ids'].to(device)
        attention_mask = d['text_attention_mask'].to(device)
        labels = d['text_labels'].to(device).long()


        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)

        correct_predictions += (preds == labels).sum()
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating", leave=False):  # 使用 tqdm 包装 data_loader
            input_ids = d['text_input_ids'].to(device)
            attention_mask = d['text_attention_mask'].to(device)
            labels = d['text_labels'].to(device).long()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += (preds == labels).sum()
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses), all_preds, all_labels

def main(epochs=10, lr=2e-5, batch_size=16):

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    from bert_models.bert_base_uncased import BERT_BASE_UNCASED_PRETRAINED_PATH, BERT_BASE_UNCASED_FTMODEL_PATH
    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH)
    
    train_loader, dev_loader, test_loader = load_data_informative(
        train_path = INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path = INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path = INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform = image_transform,
        tokenizer = tokenizer,
        max_len=160,
        batch_size=batch_size
    )
    
    print('--------------------------------')
    print('Loaded train set in length of', len(train_loader.dataset))
    print('Loaded develop set in length of', len(dev_loader.dataset))
    print('Loaded test set in length of', len(test_loader.dataset))
    print('--------------------------------')
    print('Epochs = ', epochs)
    print('LR = ', lr)
    print('Batch_size = ', batch_size)
    print('--------------------------------')
    
    model = BertForSequenceClassification.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH, num_labels=2)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
        print(f'Train loss {train_loss} accuracy {train_acc}')
        
        val_acc, val_loss, val_preds, val_labels = eval_model(model, dev_loader, device)
        print(f'Val loss {val_loss} accuracy {val_acc}')
        
        print(f'Precision: {precision_score(val_labels, val_preds, average="binary")}')
        print(f'Recall: {recall_score(val_labels, val_preds, average="binary")}')
        print(f'Confusion Matrix:\n {confusion_matrix(val_labels, val_preds)}')
        print(f'mAP: {average_precision_score(val_labels, val_preds)}')
    
    torch.save(model.state_dict(), os.path.join(BERT_BASE_UNCASED_FTMODEL_PATH, 'bert_base_uncased_model.bin'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    main(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
