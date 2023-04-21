from src.data.abstract_dataset import Dataset
import pandas as pd

DATA_PATH = "src/data/stomach/Magen_Gesamtdatensatz_REV2.csv"
DATA_INFOS_PATH = "src/data/stomach/stomach_data_infos.csv"


class StomachData(Dataset):
    def __init__(self):
        super(StomachData, self).__init__(DATA_PATH, DATA_INFOS_PATH)

    def read_csv(self):
        return (pd.read_csv(self.path, delimiter=';', skiprows=3),None)
