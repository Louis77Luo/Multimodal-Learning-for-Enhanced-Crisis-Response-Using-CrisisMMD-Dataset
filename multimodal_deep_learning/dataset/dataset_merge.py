import pandas as pd
import os

from dataset import DATASET_BASE


def merge_dataset(fold1_path, fold2_path, fold3_path):
    image_info_dict = {}

    for file_name in os.listdir(fold1_path):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(fold1_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
            for index, row in df.iterrows():
                image_id = row['image_id']
                image_info_conf = row['image_info_conf']
                image_info_dict[image_id] = image_info_conf

    for file_name in os.listdir(fold2_path):
        if file_name.endswith('.tsv'):
            file_path = os.path.join(fold2_path, file_name)
            df = pd.read_csv(file_path, sep='\t')
            df['image_info_conf'] = df['image_id'].map(image_info_dict)
            df['image_info_conf'] = df['image_info_conf'].map(lambda x: f"{x:.4f}" if pd.notnull(x) else None)
            new_file_path = os.path.join(fold3_path, file_name)
            df.to_csv(new_file_path, sep='\t', index=False)


if __name__ == '__main__':
    fold1_path = os.path.join(DATASET_BASE, 'fold1')
    fold2_path = os.path.join(DATASET_BASE, 'fold2')
    fold3_path = os.path.join(DATASET_BASE, 'fold3')

    os.makedirs(fold3_path, exist_ok=True)

    merge_dataset(fold1_path, fold2_path, fold3_path)
    print('Done')
