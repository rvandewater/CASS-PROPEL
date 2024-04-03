import logging as log
import os
from datetime import datetime


def setup_output_directory(dataset, feature_set, out_dir, offset=0):
    # Setup output directory
    # seed = seed
    # np.random.seed(seed)
    # random.seed(seed)
    now = str(datetime.now().strftime("%Y-%m-%d_T_%H-%M-%S"))
    if out_dir is None:
        # Set up an extra directory for this dataset
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'results_{dataset}{feature_set_string}_offset_{offset}_{now}'
    else:
        # Output directory already exists, make subdirectory for this run
        feature_set_string = '' if feature_set is None else f'_{"_".join(feature_set)}'
        # out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_{now}_seed_{seed}'
        out_dir = f'{out_dir}/results_{dataset}{feature_set_string}_offset_{offset}_{now}'
    log.info(f"Logging results to: {out_dir}")
    os.makedirs(f'{out_dir}', exist_ok=True)
    # os.makedirs(f'{out_dir}/data_frames', exist_ok=True)
    return out_dir, now
