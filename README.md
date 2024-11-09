# Multimodal-Learning-for-Enhanced-Crisis-Response-Using-CrisisMMD-Dataset
NLP course project. Done with CrisisMMD v2 dataset.


**For Deep Learning Part**
```
multimodal_deep_learning
│ 
├─bert_models
│  ├─bert_base_uncased
│  └─distilbert_base_uncased
├─cnn_models
│  ├─resnet18
│  └─resnet50
├─dataset
│  ├─crisismmd_datasplit_all
│  ├─data_image
│  │  ├─california_wildfires
│  │  ├─hurricane_harvey
│  │  ├─hurricane_irma
│  │  ├─hurricane_maria
│  │  ├─iraq_iran_earthquake
│  │  ├─mexico_earthquake
│  │  └─srilanka_floods
│  ├─fold1
│  ├─fold2
│  └─fold3
├─fusion_models
│  ├─bert_resnet18
│  └─bert_resnet50
└─utils
```

For python files in deep learning sction, set workspace root at multimodal_deep_learning/  and run
```python
python -m aaa.bbb.ccc --args
```
For example train the BERT model, run
```python
python -m bert_models.bert_base_uncased.bert_base_uncased --epochs 10 --lr 1e-6 --batch_size 256
```

If you have issue with internet connection to hugging face in order to fetch pretrained BERT model, just download:  
{**config.json**, **pytorch_model.bin**, **vocab.txt**} manually and move them to  
**multimodal_deep_learning/bert_models/{bert, distilbert}_base_uncased/{bert, distilbert}_base_uncased_pretrained**.
