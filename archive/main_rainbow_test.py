import os
from evaluation.MarkdownTestResult import MarkdownTestResult
from loader.Preprocessor import Preprocessor
from loader.filter_dataset import filter_acceleration
from archive.loader_archive.load_dataset import load_dataset
from models.RainbowModel import RainbowModel
from rainbow_test.create_report import (
    create_rainbow_report,
    k_fold_report_str,
    send_telegram_report,
)
from rainbow_test.k_fold_test import k_fold_cross_validation_test
from utils import settings
import random

from utils.markdown import markdown_table_str
from utils.server_manager import on_error_send_traceback, telegram_job_start_done
from utils.telegram import send_telegram


def k_fold_test(
    models: "list[RainbowModel]",
    model_nicknames: "list[str]",
    title: str,
    description: str,
    telegram: bool = False,
    k: int = 5,
) -> None:
    """
    - fits k (default 5) models for every model configuration
    - creates a report in rainbow_test/report/....md that lets you compare the results
    """
    assert len(models) == len(model_nicknames)

    raw_recordings = load_dataset(os.path.join(settings.DATA_PATH, "5-sensor-all"))
    raw_recordings = filter_acceleration(raw_recordings)
    random.seed(7930884847924791)
    random.shuffle(raw_recordings)

    preprocessor = Preprocessor()
    recordings_all = preprocessor.preprocess(raw_recordings)

    # do training and testing
    models_evaluation_results: list[list[MarkdownTestResult]] = []
    for i in range(len(models)):
        results: list[MarkdownTestResult] = k_fold_cross_validation_test(
            models[i], recordings_all, k
        )
        models_evaluation_results.append(results)
        if telegram:
            avg_correct_classification_accuracy = str(
                sum([result.correct_classification_accuracy for result in results])
                / len(results)
            )
            avg_failure_rate = str(
                sum([result.average_failure_rate for result in results]) / len(results)
            )

            send_telegram(
                "Model: "
                + model_nicknames[i]
                + " — k_fold_cross_validation_test done ("
                + str(k)
                + " models on different test_train_splits trained and tested)\navg_correct_classification_accuracy "
                + avg_correct_classification_accuracy
                + "\navg_failure_rate "
                + avg_failure_rate
            )

    # create report
    report_str = ""

    # comparison table
    comparison_table: "list[list[str | int | float]]" = [
        [""],
        ["correct_classification_acc"],
        ["avg_failure_rate"],
    ]
    for i in range(len(models)):
        comparison_table[0].append('Model "' + model_nicknames[i] + '"')
        comparison_table[1].append(
            round(
                sum(
                    [
                        i.correct_classification_accuracy
                        for i in models_evaluation_results[i]
                    ]
                )
                / len(models_evaluation_results[i]),
                ndigits=2,
            )
        )
        comparison_table[2].append(
            round(
                sum([i.average_failure_rate for i in models_evaluation_results[i]])
                / len(models_evaluation_results[i]),
                ndigits=2,
            )
        )

    report_str += markdown_table_str(comparison_table)

    # k_fold_evaluation
    for i in range(len(models)):
        model_evaluation_results = models_evaluation_results[i]
        model_nickname = model_nicknames[i]
        report_str += k_fold_report_str(
            models[i], model_nickname, model_evaluation_results
        )

    create_rainbow_report(title, description, report_str)

    if telegram:
        send_telegram_report(title, description, report_str)
