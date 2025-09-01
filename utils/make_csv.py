import pandas as pd
import os

def get_files(path):
    files = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.png'):
                files.append(os.path.join(root, file))