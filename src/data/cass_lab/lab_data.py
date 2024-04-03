from pathlib import Path

import pandas as pd
import os

from src.data.abstract_dataset import Dataset

# DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_CVKMitte.csv")
# VALIDATION_DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_Steglitz.csv")
# DATA_INFOS_PATH = os.path.abspath("src/data/esophagus/esophagus_data_infos.csv")

# DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_CVKMitte.csv")
# VALIDATION_DATA_PATH = os.path.abspath("src/data/esophagus/Datensatz_Steglitz.csv")
# DATA_INFOS_PATH = os.path.abspath("src/data/esophagus/esophagus_data_infos_updated.csv")

# DATA_PATH = Path("/sc-projects/sc-proj-cc08-cassandra/preprocessed/endpoints/joined.csv")
# VALIDATION_DATA_PATH = Path("/sc-projects/sc-proj-cc08-cassandra/preprocessed/endpoints/joined.csv")
# DATA_INFOS_PATH = os.path.abspath("src/data/cass_lab/cass_lab_data_info.csv")

DATA_PATH = Path("/sc-projects/sc-proj-cc08-cassandra/preprocessed/endpoints/joined_retro.csv")
VALIDATION_DATA_PATH = Path("/sc-projects/sc-proj-cc08-cassandra/preprocessed/endpoints/joined_retro.csv")
DATA_INFOS_PATH = os.path.abspath("src/data/cass_lab/data_info_retro.csv")

class LabData(Dataset):
    def __init__(self):
        super(LabData, self).__init__(DATA_PATH, DATA_INFOS_PATH)

    def read_csv(self):
        print(self.path)
        print(self.path.exists())
        data = pd.read_csv(self.path, delimiter=',')
        print(data)
        return (data, data)

