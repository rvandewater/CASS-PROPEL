from .esophagus.esophagus_data import EsophagusData
# from .pancreas.pancreas_data import PancreasData
from .stomach.stomach_data import StomachData
from .cass_lab.lab_data import LabData
from .cass_retro.cass_retro_elect_preop import RetroElectPreop
from .cass_retro.cass_retro_emerg_preop import RetroEmergPreop

def get_data_from_name(name):
    if name == 'esophagus':
        return EsophagusData()
    # if name == 'pancreas':
    #     return PancreasData()
    if name == 'stomach':
        return StomachData()
    if name == 'cass_lab':
        return LabData()
    if name == 'cass_preop_elect':
        return RetroElectPreop()
    if name == 'cass_preop_emerg':
        return RetroEmergPreop()
