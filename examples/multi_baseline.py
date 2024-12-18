
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
from tmu.data.tmu_dataset import TMUDataset
from tmu.data import MNIST, FashionMNIST, CIFAR10
from tmu.models.classification.distillation_classifier import DistillationClassifier
import numpy as np
import cv2

from time import time
from datetime import datetime

import argparse
import json
import logging
import os
import pdb
import ssl
import sys

ssl._create_default_https_context = ssl._create_unverified_context

SEED = 42

percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

_LOGGER = logging.getLogger(__name__)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def metrics(args, experiment_name=""):
    return dict(
        accuracy=[],
        variance=-1,
        train_time=[],
        test_time=[],
        total_time=-1,
        experiment_name=experiment_name,
        args=args
    )

def run_general_experiment(
    dataset:TMUDataset,
    num_clauses=2000,
    T=5000,
    s=10.0,
    max_included_literals=32,
    platform="CUDA",
    weighted_clauses=True,
    epochs=60,
    clause_drop_p=0.0,
    literal_drop_p=0.0,
    batch_size=256,
    experiment_name="Unnamed",
)->dict:
    """Run a general experiment with the given data and parameters.
    This will utilize the TMClassifier to train and test the data.
    Metrics returned will include the accuracy, variance, training time, testing time, and total time.


    """
    args = DotDict(
        {
            "num_clauses": num_clauses,
            "T": T,
            "s": s,
            "max_included_literals": max_included_literals,
            "platform": platform,
            "weighted_clauses": weighted_clauses,
            "epochs": epochs,
            "clause_drop_p": clause_drop_p,
            "literal_drop_p": literal_drop_p,
            "batch_size": batch_size   
        }
    )
    _LOGGER.info(f"Running {experiment_name} with {args}")
    experiment_results = metrics(args,  experiment_name)
    
    data = dataset().get()
    
    tm = DistillationClassifier(
        type_iii_feedback=False,
        number_of_clauses=num_clauses,
        T=T,
        s=s,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses,
        seed=SEED,
        clause_drop_p=clause_drop_p,
        literal_drop_p=literal_drop_p,
        batch_size=batch_size,
    )

    start_time = time()
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=_LOGGER, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                res = tm.fit(
                    data["x_train"].astype(np.uint32),
                    data["y_train"].astype(np.uint32),
                    metrics=["update_p"],
                )
                # class sums 
                _LOGGER.info(f"Res: {res}")

            experiment_results["train_time"].append(benchmark1.elapsed())
            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                prediction, class_sums = tm.predict(data["x_test"], return_class_sums=True)

                

                result = 100 * (prediction == data["y_test"]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
    experiment_results["variance"] = np.var(experiment_results["accuracy"])
    return experiment_results


def run_kd_experiment(
    dataset:TMUDataset,
    teacher_num_clauses=2000,
    teacher_T=5000,
    teacher_s=10.0,
    student_num_clauses=80,
    student_T=5000,
    student_s=10.0,
    teacher_multiplier=1.0,
    shuffle=False,
    max_included_literals=32,
    platform="CUDA",
    weighted_clauses=True,
    epochs=60,
    clause_drop_p=0.0,
    literal_drop_p=0.0,
    batch_size=256,
    experiment_name="Unnamed",
)->dict:
    """Run a general experiment with the given data and parameters.
    This will utilize the TMClassifier to train and test the data.
    Metrics returned will include the accuracy, variance, training time, testing time, and total time.
    """

    assert teacher_num_clauses > student_num_clauses, "Teacher must have more clauses than student"

    args = DotDict(
        {
            "teacher_num_clauses": teacher_num_clauses,
            "teacher_T": teacher_T,
            "teacher_s": teacher_s,
            "student_num_clauses": student_num_clauses,
            "student_T": student_T,
            "student_s": student_s,

            "max_included_literals": max_included_literals,
            "platform": platform,
            "weighted_clauses": weighted_clauses,
            "epochs": epochs,
            "clause_drop_p": clause_drop_p,
            "literal_drop_p": literal_drop_p,
            "batch_size": batch_size   
        }
    )
    _LOGGER.info(f"Running {experiment_name} with {args}")
    experiment_results = metrics(args,  experiment_name)
    
    data = dataset().get()
    
    teacher = DistillationClassifier(
        type_iii_feedback=False,
        number_of_clauses=teacher_num_clauses,
        T=teacher_T,
        s=teacher_s,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses,
        seed=SEED,
        clause_drop_p=clause_drop_p,
        literal_drop_p=literal_drop_p,
        batch_size=batch_size,
    )

    student = DistillationClassifier(
        type_iii_feedback=False,
        number_of_clauses=student_num_clauses,
        T=student_T,
        s=student_s,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses,
        seed=SEED,
        clause_drop_p=clause_drop_p,
        literal_drop_p=literal_drop_p,
        batch_size=batch_size,
    )

    basic = TMClassifier(
        number_of_clauses=student_num_clauses,
        T=student_T,
        s=student_s,
        max_included_literals=max_included_literals,
        platform=platform,
        weighted_clauses=weighted_clauses,
        seed=SEED,
        clause_drop_p=clause_drop_p,
        literal_drop_p=literal_drop_p,
        batch_size=batch_size,
    )

    start_time = time()
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=_LOGGER, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                # train teacher
                teacher_res, class_sums, class_sums_not = teacher.fit(
                    data["x_train"].astype(np.uint32),
                    data["y_train"].astype(np.uint32),
                    metrics=["update_p"],
                    shuffle=shuffle,
                )
                
                # then fit student
                student_res, _, _ = student.fit(
                    data["x_train"].astype(np.uint32),
                    data["y_train"].astype(np.uint32),
                    teacher_multiplier=teacher_multiplier,
                    teacher_sums=class_sums,
                    teacher_sums_not=class_sums_not,
                    metrics=["update_p"],
                    shuffle=shuffle,
                )

            experiment_results["train_time"].append(benchmark1.elapsed())
            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                prediction, class_sums = student.predict(data["x_test"], return_class_sums=True)
                result = 100 * (prediction == data["y_test"]).mean()
                experiment_results["accuracy"].append(result)

            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
    experiment_results["variance"] = np.var(experiment_results["accuracy"])
    return experiment_results


if __name__ == "__main__":
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default=os.path.join("logs",
                        f"multi-baseline-{current_time}.log"), type=str)
    parser.add_argument("--no-log", action="store_true", default=False, help="Disable logging to file")

    args = parser.parse_args()

    if not args.no_log:
        if not os.path.exists("logs"):
            os.makedirs("logs")
        if os.path.exists(args.log_file):
            os.remove(args.log_file)
        if os.path.exists("latest.log"):
            os.remove("latest.log")

    # Set the desired log level here
    _LOGGER.setLevel(logging.DEBUG)

    # Create a formatter to define the log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S %Z')

    # Create a stream handler to print logs to the console
    #console_handler = logging.StreamHandler()
    #console_handler.setLevel(logging.INFO)
    #console_handler.setFormatter(formatter)
    #_LOGGER.addHandler(console_handler)

    if not args.no_log:
        # Create a file handler to write logs to a file
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        _LOGGER.addHandler(file_handler)

        lfile_handler = logging.FileHandler("latest.log")
        lfile_handler.setLevel(logging.INFO)
        lfile_handler.setFormatter(formatter)
        _LOGGER.addHandler(lfile_handler)
        _LOGGER.info(f"Logging to {args.log_file} and latest.log")


    results = []
    
    kd_experiments = [
        {"dataset":MNIST, "experiment_name": "MNIST", "teacher_num_clauses": 2000, "teacher_T": 5000, "teacher_s": 10.0, "student_num_clauses": 80, "student_T": 5000, "student_s": 10.0, "teacher_multiplier": 1.0, "epochs": 60},
    ]

    for ex in kd_experiments:
        result = run_kd_experiment(**ex)
        _LOGGER.info(result)
        # save results
        if not args.no_log:
            with open(os.path.join("logs", f"results-{current_time}.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open("latest.json", "w") as f:
                json.dump(results, f, indent=4)

    sys.exit(0)

    general_experiments = [
        {"dataset":MNIST, "experiment_name": "MNIST", "num_clauses": 80, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
        {"dataset":MNIST, "experiment_name": "MNIST", "num_clauses": 800, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
        {"dataset":MNIST, "experiment_name": "MNIST", "num_clauses": 2000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
        {"dataset":CIFAR10, "experiment_name": "CIFAR10","num_clauses": 10000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60},
        {"dataset":CIFAR10,"experiment_name": "CIFAR10", "num_clauses": 15000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60},
        {"dataset":CIFAR10, "experiment_name": "CIFAR10", "num_clauses": 20000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60},
        {"dataset":FashionMNIST, "experiment_name": "FashionMNIST", "num_clauses": 100, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
        {"dataset":FashionMNIST, "experiment_name": "FashionMNIST", "num_clauses": 1000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
        {"dataset":FashionMNIST, "experiment_name": "FashionMNIST", "num_clauses": 2000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60},
    ]

    for ex in general_experiments:
        result = run_general_experiment(**ex)
        results.append(result)
        # save results
        if not args.no_log:
            with open(os.path.join("logs", f"results-{current_time}.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open("latest.json", "w") as f:
                json.dump(results, f, indent=4)

    # print results
    for result in results:
        _LOGGER.info(result)

