import json
import math
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from math import e

def aggregate_results(
        log_dir: Path,
        results_files=["roc_auc", "prc_auc", "f1_score", "recall", "precision", "recall", "avg_precision", "balanced_accuracy",
                       "mcc", ],
):
    """
    Aggregate results from a log directory.
    Args:
        log_dir: Log directory stub.
        results_files: Metric to aggregate.
    """
    for results_file in results_files:
        runs = []
        for run_instance in log_dir.iterdir():
            if run_instance.is_dir():
                iteration_result = Path(run_instance / "data_frames" / f"{results_file}.csv")
                if iteration_result.exists():
                    result = pd.read_csv(open(iteration_result), index_col=0)
                    # result = result.apply(lambda x: math.pow(x,e))
                    runs.append(result)
        mean = pd.concat(runs).groupby(level=0).mean()
        stdev = pd.concat(runs).groupby(level=0).std()
        sem = pd.concat(runs).groupby(level=0).sem()
        stdev = stdev.add_suffix("_std")

        ci_hi = mean + sem.apply(lambda x: x * 1.96)
        ci_lo = mean - sem.apply(lambda x: x * 1.96)

        mean = mean.add_suffix("_mean")
        ci_hi = ci_hi.add_suffix("_ci_95_hi")
        ci_lo = ci_lo.add_suffix("_ci_95_lo")
        final = pd.concat([mean, stdev, ci_hi, ci_lo], axis=1)
        final = final.reindex(sorted(final.columns), axis=1)
        final.to_csv(log_dir / f"{results_file}_accumulated.csv")
        # result_frames = result_frames.append(final)


aggregate_results(Path(r"./final_results_internal/"))

def aggregate_feature_importances(
        log_dir: Path,
        model_names=["DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier","LogisticRegression","SVC","LinearSVC"]
):
    """
    Aggregate results from a log directory.
    Args:
        log_dir: Log directory stub.
        results_files: Metric to aggregate.
    """
    for model_name in model_names:
        runs = []
        for run_instance in log_dir.iterdir():
            if run_instance.is_dir():
                for endpoint in run_instance.iterdir():
                    if endpoint.is_dir():
                        iteration_result = Path(endpoint / f"{model_name}_feature_importance.csv")
                        if iteration_result.exists():
                            result = pd.read_csv(open(iteration_result),index_col=0)
                            if model_name == "LogisticRegression":
                                result = result.apply(lambda x: math.pow(e,x), axis=0, result_type="broadcast")
                            runs.append(result)

                            runs.append(pd.read_csv(open(iteration_result), index_col=0))
        concatenated = pd.concat(runs).groupby(level=0).sum()
        concatenated = concatenated.transpose()
        concatenated.rename(columns={0: "Accumulated Scaled Importance"}, inplace=True)
        # concatenated = concatenated - concatenated.min()
        # concatenated = concatenated / concatenated.max()
        concatenated.sort_values(by="Accumulated Scaled Importance", ascending=False, inplace=True)
        # concatenated.plot.barh(width=20)
        # plt.show()
        concatenated.to_csv(log_dir / f"{model_name}_importance_accumulated.csv")


#aggregate_feature_importances(Path(r"C:\Users\Robin\Downloads\feature_importances_corrected_"))
