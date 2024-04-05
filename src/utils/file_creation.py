import logging as log
import os
from datetime import datetime


def setup_output_directory(dataset, feature_set, out_dir, offset=None):
    # Setup output directory
    now = str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))
    if out_dir is None:
        # Set up an extra directory for this dataset
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        if offset is not None:
            out_dir = f'results_{dataset}{feature_set_string}_offset_{offset}_{now}'
        else:
            out_dir = f'results_{dataset}{feature_set_string}_{now}'
    else:
        # Output directory already exists, make subdirectory for this run
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        if offset is not None:
            out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_offset_{offset}_{now}'
        else:
            out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{now}'
    log.info(f"Logging results to: {out_dir}")
    os.makedirs(f'{out_dir}', exist_ok=True)
    return out_dir, now
