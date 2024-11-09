from utils import device
from bert_models.bert_base_uncased import BERT_BASE_UNCASED_FTMODEL_PATH

import torch
from transformers import BertForSequenceClassification, BertConfig
import os


class BertEncoder:
    def __init__(self, model_path, ft_model_path, device='cpu'):
        config = BertConfig.from_pretrained(model_path, num_labels=2, output_hidden_states=True)
        self.model = BertForSequenceClassification.from_pretrained(model_path, config=config)
        self.model.load_state_dict(torch.load(os.path.join(ft_model_path, 'bert_base_uncased_model.bin'), weights_only=True))
        self.model.to(device)
        self.model.eval()

        self.device = device

    def encode(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            hidden_states = outputs.hidden_states
            cls_embedding = hidden_states[-1][:, 0, :]  # 使用最后一层隐藏状态的[CLS] token
        return cls_embedding



if __name__ == '__main__':
    from dataset import (INFORMATIVE_DATASET_TRAIN_MULTI_PATH,
                         INFORMATIVE_DATASET_DEV_MULTI_PATH,
                         INFORMATIVE_DATASET_TEST_MULTI_PATH)
    from utils import device
    from utils.load_data import load_data_informative
    from bert_models.bert_base_uncased import BERT_BASE_UNCASED_PRETRAINED_PATH, BERT_BASE_UNCASED_FTMODEL_PATH
    from torchvision import transforms
    from transformers import BertTokenizer

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

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

    encoder = BertEncoder(model_path=BERT_BASE_UNCASED_PRETRAINED_PATH,
                          ft_model_path=BERT_BASE_UNCASED_FTMODEL_PATH,
                          device=device)

    for data in test_loader:
        input_ids = data['text_input_ids']
        attention_mask = data['text_attention_mask']
        cls_embeddings = encoder.encode(input_ids, attention_mask)
        print(cls_embeddings.size())
        break
