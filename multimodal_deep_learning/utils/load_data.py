from dataset import (DATASET_BASE,
                     INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)
import torch
import torchvision
from transformers import BertTokenizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL

import re
import os


def clean_tweet(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize
    words = text.split()
    return ' '.join(words)


class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, image_transform, tokenizer, max_len):
        self.data = pd.read_csv(file_path, sep='\t')
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = row['tweet_text']
        text = clean_tweet(text)
        text_label = row['label_text']

        text_encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        image_path = row['image']
        image = PIL.Image.open(os.path.join(DATASET_BASE, image_path)).convert('RGB')
        image = self.image_transform(image)
        
        image_label = row['label_image']
        image_info_conf = row['image_info_conf']

        return {
            'image_path': image_path,
            'image': image,
            'image_label': torch.tensor(1 if image_label == 'informative' else 0, dtype=torch.float),
            'image_info_conf': torch.tensor(image_info_conf if image_label == 'informative' else -image_info_conf,
                                            dtype=torch.float),
            'image_info_conf_norm': (torch.tensor(image_info_conf if image_label == 'informative' else -image_info_conf,
                                                  dtype=torch.float) + 1) / 2.0,

            'text': text,
            'text_input_ids': text_encoding['input_ids'].flatten(),
            'text_attention_mask': text_encoding['attention_mask'].flatten(),
            'text_labels': torch.tensor(1 if text_label == 'informative' else 0, dtype=torch.float)
        }


def load_data_informative(train_path, dev_path, test_path, image_transform, tokenizer, max_len, batch_size):
    train_dataset = TweetDataset(file_path=train_path,
                                 image_transform=image_transform,
                                 tokenizer=tokenizer,
                                 max_len=max_len)
    dev_dataset = TweetDataset(file_path=dev_path,
                               image_transform=image_transform,
                               tokenizer=tokenizer,
                               max_len=max_len)
    test_dataset = TweetDataset(file_path=test_path,
                                image_transform=image_transform,
                                tokenizer=tokenizer,
                                max_len=max_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, dev_loader, test_loader


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
        batch_size=4
    )

    print(len(train_loader))
    print(len(train_loader.dataset))
    print(len(dev_loader.dataset))
    print(len(test_loader.dataset))

    for d in dev_loader:
        print(d['image_path'])
        print(d['image'].size())
        print(d['image_label'])
        print(d['image_info_conf'])
        print(d['image_info_conf_norm'])

        print(d['text'])
        print(d['text_input_ids'].size())
        print(d['text_attention_mask'].size())
        print(d['text_labels'])

        plt.figure(figsize=(8, 8))
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.imshow(np.array(255 * d['image'][i, :, :, :], dtype=np.uint8, copy=True).transpose((1, 2, 0)))
            plt.title(d['text'][i][:30] + '...')
        plt.show()

        break
