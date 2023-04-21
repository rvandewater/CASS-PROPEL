import pandas as pd
import os

from src.data.abstract_dataset import Dataset

# DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_CVKMitte.csv")
# VALIDATION_DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_Steglitz.csv")
# DATA_INFOS_PATH = os.path.abspath("src/data/esophagus/esophagus_data_infos.csv")

# DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_CVKMitte.csv")
# VALIDATION_DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_Steglitz.csv")
# DATA_INFOS_PATH = os.path.abspath("src/data/esophagus/esophagus_data_infos_updated.csv")

DATA_PATH = os.path.abspath("src/data/esophagus/ESOCVK.csv")
VALIDATION_DATA_PATH = os.path.abspath("src/data/esophagus/ESOSTEG.csv")
DATA_INFOS_PATH = os.path.abspath("src/data/esophagus/esophagus_data_infos.csv")

class EsophagusData(Dataset):
    def __init__(self):
        super(EsophagusData, self).__init__(DATA_PATH, DATA_INFOS_PATH, VALIDATION_DATA_PATH)

    def read_csv(self):
        return (pd.read_csv(self.path, delimiter=','), pd.read_csv(self.validation_data_path))

