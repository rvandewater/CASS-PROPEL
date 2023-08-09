from .esophagus.esophagus_data import EsophagusData
# from .pancreas.pancreas_data import PancreasData
from .stomach.stomach_data import StomachData


def get_data_from_name(name):
    if name == 'esophagus':
        return EsophagusData()
    # if name == 'pancreas':
    #     return PancreasData()
    if name == 'stomach':
        return StomachData()
