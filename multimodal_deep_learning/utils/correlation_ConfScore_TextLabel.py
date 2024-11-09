import torch
import torchvision
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import PIL
import os
import re
from tqdm import tqdm

from utils.load_data import load_data_informative
from dataset import (DATASET_BASE,
                     INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                     INFORMATIVE_DATASET_DEV_MULTI_PATH,
                     INFORMATIVE_DATASET_TEST_MULTI_PATH)


def binarize_image_info_conf(image_info_conf):
    return [1 if x > 0 else 0 for x in image_info_conf]


# Pearson correlation coefficient: 0.44252961720394524, p-value: 0.0
# Point-Biserial correlation coefficient: 0.44252961720394524, p-value: 0.0
# Matching elements: 9601, Total elements: 13608, Matching ratio: 0.7055408583186361
if __name__ == '__main__':
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 加载分词器
    from bert_models.bert_base_uncased import BERT_BASE_UNCASED_PRETRAINED_PATH

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED_PRETRAINED_PATH)

    # 加载数据加载器
    train_loader, _, _ = load_data_informative(
        train_path=INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
        dev_path=INFORMATIVE_DATASET_DEV_MULTI_PATH,
        test_path=INFORMATIVE_DATASET_TEST_MULTI_PATH,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_len=160,
        batch_size=4
    )

    # 收集 image_info_conf_norm 和 text_labels
    image_info_conf_norms = []
    text_labels = []

    for batch in tqdm(train_loader):
        image_info_conf_norms.extend(batch['image_info_conf'].numpy())
        text_labels.extend(batch['text_labels'].numpy())

    # 计算皮尔逊相关系数--------------------------------------------------------------
    correlation, p_value = pearsonr(image_info_conf_norms, text_labels)
    print(f"Pearson correlation coefficient: {correlation}, p-value: {p_value}")

    # pointbiserialr 相关系数---------------------------------------------------------
    from scipy.stats import pointbiserialr

    # 假设 image_info_conf_norms 和 text_labels 已经收集好
    correlation, p_value = pointbiserialr(image_info_conf_norms, text_labels)
    print(f"Point-Biserial correlation coefficient: {correlation}, p-value: {p_value}")

    # Binarize image_info_conf_norms based on threshold 0------------------------------
    binarized_image_info_conf_norms = binarize_image_info_conf(image_info_conf_norms)

    # Calculate the ratio of matching elements to total elements
    matching_elements = sum([1 for i in range(len(binarized_image_info_conf_norms)) if
                             binarized_image_info_conf_norms[i] == text_labels[i]])
    total_elements = len(binarized_image_info_conf_norms)
    matching_ratio = matching_elements / total_elements

    print(f"Matching elements: {matching_elements}, Total elements: {total_elements}, Matching ratio: {matching_ratio}")
