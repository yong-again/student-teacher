"""
make_csv.py
make csv file for model train and validation.
"""

import pandas as pd
import os
import glob
from pydantic import BaseModel
from typing import List, Dict, Union


class Settings(BaseModel):
    """MVTec AD 데이터셋 관련 설정을 관리하는 클래스"""
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = 'data'
    mvtec_folder: str = 'mvtec_AD'
    csv_folder: str = 'csv'


class MVTecDataManager:
    """MVTec AD 데이터셋의 경로, 카테고리 및 데이터프레임 생성을 관리하는 클래스"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.mvtec_path = os.path.join(self.settings.root_dir, self.settings.data_dir, self.settings.mvtec_folder)
        self.save_path = os.path.join(self.settings.root_dir, self.settings.data_dir, self.settings.csv_folder)
        self.categories = self._get_categories()

    def _get_categories(self) -> List[str]:
        """데이터셋 경로에서 카테고리 목록을 가져옵니다."""
        categories = [folder for folder in os.listdir(self.mvtec_path) if not folder.startswith('.')]
        return sorted(categories)

    def _process_data(self, file_pattern: str, split_type: str, has_anomaly_label: bool = False) -> pd.DataFrame:
        """주어진 패턴으로 이미지 경로를 처리하여 데이터프레임을 생성합니다."""
        df_list = []
        for category_name in self.categories:
            category_path = os.path.join(self.mvtec_path, category_name)
            file_paths = glob.glob(os.path.join(category_path, file_pattern))
            file_paths.sort()

            if not file_paths:
                continue

            df = pd.DataFrame(file_paths, columns=['filepath'])
            df['category'] = category_name
            df['split'] = split_type

            if has_anomaly_label:
                df['anomaly'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[0])
                df['filename'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[1])
            else:
                df['filename'] = df['filepath'].apply(lambda x: x.split(f'/{split_type}/')[1].split('/')[0])

            df_list.append(df)

        if not df_list:
            return pd.DataFrame()

        combined_df = pd.concat(df_list, ignore_index=True)
        return combined_df.drop(columns=['filepath'])

    def create_and_save_dataframes(self):
        """데이터프레임을 생성하고 CSV 파일로 저장합니다."""
        data_patterns = {
            'train': ('train/good/*.png', False),
            'test': ('test/*/*.png', True),
            'ground_truth': ('ground_truth/*/*.png', True),
        }

        dataframes_to_save: Dict[str, pd.DataFrame] = {}

        for name, (pattern, has_anomaly) in data_patterns.items():
            df = self._process_data(pattern, name, has_anomaly)
            dataframes_to_save[name] = df

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for name, df in dataframes_to_save.items():
            if not df.empty:
                df.to_csv(os.path.join(self.save_path, f'{name}.csv'), index=False)
                print(f"✅ Successfully saved {name}.csv")


if __name__ == '__main__':
    settings = Settings()
    manager = MVTecDataManager(settings)
    manager.create_and_save_dataframes()