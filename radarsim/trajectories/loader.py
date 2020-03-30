import os
import pandas as pd


def get_file_list():
    files = os.listdir('resources')
    return files


def load_trajectory(name):
    pd.read_csv()