# CASS-PROPEL (Clinical ASSist - Patient Risk/Outcome Prediction and EvaLuation)

**Complete evaluation of traditional "SK-learn like" machine learning models for post-operative complications**

This repository serves to evaluate the performance of traditional machine learning models available from [Scikit-learn](https://github.com/scikit-learn/scikit-learn) 
for the prediction of post-operative complications. Although the focus is on the prediction of post-operative 
complications, the code can be used and adjusted for other use cases.

The code was used to generate the results for the following papers:
* R. van de Water et al., ‘Combining Hospital-grade Clinical Data and Wearable Vital Sign Monitoring to Predict Surgical Complications’, presented at the ICLR 2024 Workshop on Learning from Time Series For Health, Mar. 2024. Accessed: Apr. 03, 2024. [Online]. Available: https://openreview.net/forum?id=EzNGSRPGa7
* R. van de Water et al., ‘Combining Time Series Modalities to Create Endpoint-driven Patient Records’, presented at the ICLR 2024 Workshop on Data-centric Machine Learning Research (DMLR): Harnessing Momentum for Science, Mar. 2024. Accessed: Apr. 03, 2024. [Online]. Available: https://openreview.net/forum?id=0NZOSSBZCi

An older version of the code (https://github.com/HPI-CH/PROPEL) was used to generate the results for the following papers:
* [Pfitzner, B., Chromik, J., Brabender, R., Fischer, E., Kromer, A., Winter, A., Moosburner, S., Sauer, I. M., Malinka, T., Pratschke, J., Arnrich, B., & Maurer, M. M. (2021). Perioperative Risk Assessment in Pancreatic Surgery Using Machine Learning. Annual International Conference of the IEEE Engineering in Medicine and Biology Society. IEEE Engineering in Medicine and Biology Society. Annual International Conference, 2021, 2211–2214. https://doi.org/10.1109/EMBC46164.2021.9630897](https://doi.org/10.1109/EMBC46164.2021.9630897)
* (UNDER REVIEW) Winter, A., van de Water, R., Pfitzner, B., Riepe, C., Ahlborn, R., Faraj, L., Krenzien, F., Dobrindt, E., Raakow, J., Sauer, I., Arnrich, B., Beyer, K., Denecke, C., Pratschke, J., Maurer, M. (2023). Advancing Preoperative Outcome Prediction: A Comparative Analysis of Machine Learning and ISEG Risk Score for Predicting 90-Day Mortality after Esophagectomy
________
## Getting started
Clone repository \
```git clone https://github.com/HPI-CH/PROPEL.git``` \
```cd PROPEL```

Create virtual environment based on Python 3.8 and activate it (e.g., using Conda) \
```conda create --name {name_of_your_choice} python=3.8``` \
```conda activate {name_of_your_choice}```

Install requirements \
```pip install -r requirements.txt```

### Adding new datasets
Create python package ```{dataset_name}``` in ```./src/data/``` (add empty ```__init__.py``` file) \
Define dataset by creating:
* ```./src/data/{dataset_name}/{dataset_name}_data.py```
  * This file should specify the locations of the data csv files (training and optional external validation cohorts)
  * It should also create a new class inheriting from ```Dataset``` in ```./src/data/abstract_dataset.py``` and implementing the `read_csv()` method which holds the code for reading the dataset into a pandas dataframe.
* ```./src/data/{dataset_name}/{dataset_name}_data_infos.csv```
  * This file should contain the following columns:
    * `column_name`: name of the feature/column in the data csv file.
    * `endpoint`: boolean indicating whether the feature is the target variable.
    * `data_type`: type of the feature - one of `N`, `C`, `B` for numerical, categorical, binary features.
    * `input time`: the time of collection of the feature - one of `pre`, `intra`, `post`, `dyn` for pre-, intra-, post-operative or dynamic (time-series) features.
    * `drop`: boolean indicating if the feature should be dropped from the dataset (before any algorithmic feature elimination).
  * This information is necessary for the data loading and preprocessing steps.
_______
## Usage

The entrypoint for the evaluation is the `complete_evaluation.py` script. It takes the following arguments:
* `dataset`: name of the dataset to evaluate, needs to have an associated package in `src/data`
* `--feature_set`, `-f`: if the dataset is split into pre-, intra- and post-operative features, a subset can be defined here
* `--external_testset`, `-e`: whether to use an external test set (or alternatively use train-test split of the training data)
* `--out_dir`, `-o`: base directory to store the results in
* `--no_features_dropped`, `-nfd`: deactivates dropping predefined features in dataframe
* `--no_feature_selection`, `-nfs`: deactivates feature selection in pipeline
* `--cv_splits`, `-cv`: number of cross_validation splits; 1 denotes LOO-CV
* `--shap_eval`, `-sh`: if true, shap plots will be generated (increases runtime a lot)
* `--test_fraction`, `-b`: fraction of data to use for test set (unused if `--external_testset` is `True`)
* `--balancing_option`, `-b`: technique to deal with imbalanced data (one of `class_weight`, `random_oversampling`, `SMOTE`, `ADASYN`, `none`)
* `--drop_missing_value`, `-dr`: Drop rows with x% of columns having missing values
* `--missing_threshold`, `-mt`: Drop columns with x% of rows having missing values
* `--correlation_threshold`, `-ct`: Drop columns with correlation above threshold
* `--data_exploration`, `-ex`: if true, an html file will be generated showing statistics of the parsed dataset
* `--seed`, `-s`: random state for initialisations

The script will then perform a cross-validation and hyperparameter optimisation of all models and the
respective parameter options defined in `src/models.py` and store the results in the specified output directory. 
It will include one sub-directory per endpoint holding plots and csv files for validation and testing for individual model types, as well as aggregated across models.
The text file `best_parameters.txt` in the output directory will contain the best parameters for each model type and endpoint.

--------
## Authors
* [Bjarne Pfitzner](https://github.com/BjarnePfitzner)
* [Robin van de Water](https://github.com/rvandewater)

--------
## Project Organization

    ├── LICENSE                 <- MIT License
    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                              generated with `pip freeze > requirements.txt`
    │
    ├── complete_evaluation.py  <- Main script to run the ML evaluation. 
    └── src                     <- Additional source code for use in this project.
        ├── __init__.py             <- Makes src a Python module
        │
        ├── data                    <- Scripts to process data
        │   ├── __init__.py             <- Makes data a Python module
        │   ├── esophagus               <- Folder for the esophagus dataset (exemplary for other datasets)
        │   └── abstract_dataset.py     <- Defines the python superclass and functionality for all datasets
        │
        ├── evaluate.py             <- Script to evaluate a single model type
        ├── models.py               <- Definition of the models to evaluate and their hyperparameter space
        ├── preprocess.py           <- Script to preprocess the data
        │
        ├── utils                   <- Utility code
        │   ├── __init__.py             <- Makes utils a Python module
        │   ├── feature_selector.py     <- Code for basic feature selection based on correlation, missing valies etc.
        │   ├── metrics.py              <- Provides function to compute all metrics from model output
        │   └── plot.py                 <- Holds many functions for plotting results
        │
        └── test.py                 <- Script for the final evaluation on the test set
