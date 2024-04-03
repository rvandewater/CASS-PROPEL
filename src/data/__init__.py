from .esophagus.esophagus_data import EsophagusData
# from .pancreas.pancreas_data import PancreasData
from .stomach.stomach_data import StomachData
from .cass_prop.lab_data import CassPropData
from .cass_retro.cass_retro_elect_preop import RetroElectPreop
from .cass_retro.cass_retro_emerg_preop import RetroEmergPreop

def get_data_from_name(name, offset=12):
    if name == 'esophagus':
        return EsophagusData()
    # if name == 'pancreas':
    #     return PancreasData()
    if name == 'stomach':
        return StomachData()
    if name == 'cass_prop':
        return CassPropData(offset)
    if name == 'cass_preop_elect':
        return RetroElectPreop()
    if name == 'cass_preop_emerg':
        return RetroEmergPreop()
