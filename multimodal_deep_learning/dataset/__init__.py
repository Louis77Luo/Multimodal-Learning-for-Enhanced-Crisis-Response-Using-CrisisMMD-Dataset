import os

DATASET_BASE = os.path.abspath(os.path.join(__file__, ".."))

INFORMATIVE_DATASET_TRAIN_PATH = os.path.join(DATASET_BASE, 'crisismmd_datasplit_all', 'task_informative_text_img_train.tsv')
INFORMATIVE_DATASET_TEST_PATH = os.path.join(DATASET_BASE, 'crisismmd_datasplit_all', 'task_informative_text_img_test.tsv')
INFORMATIVE_DATASET_DEV_PATH = os.path.join(DATASET_BASE, 'crisismmd_datasplit_all', 'task_informative_text_img_dev.tsv')

INFORMATIVE_DATASET_TRAIN_MULTI_PATH = os.path.join(DATASET_BASE, 'fold3', 'task_informative_text_img_multi_train.tsv')
INFORMATIVE_DATASET_TEST_MULTI_PATH = os.path.join(DATASET_BASE, 'fold3', 'task_informative_text_img_multi_test.tsv')
INFORMATIVE_DATASET_DEV_MULTI_PATH = os.path.join(DATASET_BASE, 'fold3', 'task_informative_text_img_multi_dev.tsv')