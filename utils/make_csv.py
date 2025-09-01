"""
make_csv.py
make csv file for model train and validation.
"""


import os
import pandas as pd
import glob

def get_mvtec_path(root_dir):
    """Returns the path to the MVTec AD dataset directory."""
    data_dir = os.path.join(root_dir, 'data')
    return os.path.join(data_dir, 'mvtec_AD')


def get_categories(path):
    """Extracts and sorts categories from the dataset directory."""
    categories = [folder for folder in os.listdir(path) if not folder.startswith('.')]
    return sorted(categories)


def process_mvtec_data(path, categories, file_pattern, split_type, has_anomaly_label=False):
    """
    Processes image paths for a given data split and returns a pandas DataFrame.
    """
    df_list = []
    for category_name in categories:
        category_path = os.path.join(path, category_name)
        file_paths = glob.glob(os.path.join(category_path, file_pattern))
        file_paths.sort()

        if not file_paths:
            continue

        df = pd.DataFrame(file_paths, columns=['filepath'])
        df['category'] = category_name
        df['split'] = split_type

        # Extract anomaly and filename if applicabler
        if has_anomaly_label:
            df['anomaly'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[0])
            df['filename'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[1])
            df['label'] = 1
        else:
            df['filename'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[1])
            df['label'] = 0

        df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df.drop(columns=['filepath'])


def save_dataframes(dfs, save_dir):
    """Saves DataFrames to CSV files in the specified directory."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for name, df in dfs.items():
        if not df.empty:
            df.to_csv(os.path.join(save_dir, f'{name}.csv'), index=False)
            print(f"Successfully saved {name}.csv")


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mvtec_path = get_mvtec_path(root_dir)
    categories = get_categories(mvtec_path)

    # Define file patterns
    train_reg = 'train/good/*.png'
    test_reg = 'test/*/*.png'
    gt_reg = 'ground_truth/*/*.png'

    # Process and create DataFrames
    df_train = process_mvtec_data(mvtec_path, categories, train_reg, 'train')
    df_test = process_mvtec_data(mvtec_path, categories, test_reg, 'test', has_anomaly_label=True)
    df_gt = process_mvtec_data(mvtec_path, categories, gt_reg, 'ground_truth', has_anomaly_label=True)

    # Save DataFrames
    save_path = os.path.join(root_dir, 'data', 'csv')
    dataframes_to_save = {
        'train': df_train,
        'test': df_test,
        'ground_truth': df_gt
    }
    save_dataframes(dataframes_to_save, save_path)