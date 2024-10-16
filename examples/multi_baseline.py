
import logging
import argparse
from tmu.models.classification.vanilla_classifier import TMClassifier
from tmu.tools import BenchmarkTimer
from tmu.util.cuda_profiler import CudaProfiler
import numpy as np
from keras.datasets import cifar10, cifar100
import cv2
from datetime import datetime
import os
import numpy as np

import json
from tmu.data import MNIST, FashionMNIST, CIFAR10

import numpy as np
from time import time

import ssl
import pdb
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
        train_time=[],
        test_time=[],
        total_time=-1,
        experiment_name=experiment_name,
        args=args
    )

def run_cifar10(
    num_clauses=2000,
    T=5000 // 100,
    s=10.0,
    max_included_literals=32,
    device="CUDA",
    weighted_clauses=True,
    epochs=60,
    type_i_ii_ratio=1.0,
    clause_drop_p=0.0,
    literal_drop_p=0.0,
    batch_size=512,
):
    
    args = DotDict(
        {
            "num_clauses": num_clauses,
            "T": T,
            "s": s,
            "max_included_literals": max_included_literals,
            "device": device,
            "weighted_clauses": weighted_clauses,
            "epochs": epochs,
            "type_i_ii_ratio": type_i_ii_ratio,
            "clause_drop_p": clause_drop_p,
            "literal_drop_p": literal_drop_p,
            "batch_size": batch_size
        }
    )
    _LOGGER.info(f"Running CIFAR10 with {args}")
    experiment_results = metrics(args, "CIFAR10")

    # switch between using keras dataset and tmu dataset

    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    #data = CIFAR10().get()
    #(X_train, Y_train), (X_test, Y_test) = (data["x_train"], data["y_train"]), (data["x_test"], data["y_test"])

    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    Y_train = Y_train
    Y_test = Y_test

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            X_train[i, :, :, j] = cv2.adaptiveThreshold(
                X_train[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
                # cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            )

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
            X_test[i, :, :, j] = cv2.adaptiveThreshold(
                X_test[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
                # cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            )

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        type_i_ii_ratio=args.type_i_ii_ratio,
        seed=SEED,
        clause_drop_p=args.clause_drop_p,
        literal_drop_p=args.literal_drop_p,
        batch_size=args.batch_size,
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    start_time = time()
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=_LOGGER, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)
            experiment_results["train_time"].append(benchmark1.elapsed())

            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(
                f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                f"Testing Time: {benchmark2.elapsed():.2f}s"
            )

        if args.device == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)
    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
    return experiment_results



def run_cifar100(
    num_clauses=2000,
    T=5000 // 100,
    s=10.0,
    max_included_literals=32,
    device="CUDA",
    weighted_clauses=True,
    epochs=60,
    type_i_ii_ratio=1.0,
    clause_drop_p=0.0,
    literal_drop_p=0.0,
    batch_size=100,
):
    args = DotDict(
        {
            "num_clauses": num_clauses,
            "T": T,
            "s": s,
            "max_included_literals": max_included_literals,
            "device": device,
            "weighted_clauses": weighted_clauses,
            "epochs": epochs,
            "type_i_ii_ratio": type_i_ii_ratio,
            "clause_drop_p": clause_drop_p,
            "literal_drop_p": literal_drop_p,
            "batch_size": batch_size
        }
    )
    _LOGGER.info(f"Running CIFAR10 with {args}")
    experiment_results = metrics(args, "CIFAR100")

    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data() 

    X_train = np.copy(X_train)
    X_test = np.copy(X_test)
    Y_train = Y_train
    Y_test = Y_test

    Y_train = Y_train.reshape(Y_train.shape[0])
    Y_test = Y_test.reshape(Y_test.shape[0])

    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[3]):
            X_train[i, :, :, j] = cv2.adaptiveThreshold(
                X_train[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
                # cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            )

    for i in range(X_test.shape[0]):
        for j in range(X_test.shape[3]):
            X_test[i, :, :, j] = cv2.adaptiveThreshold(
                X_test[i, :, :, j],
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
                # cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
            )

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        type_i_ii_ratio=args.type_i_ii_ratio,
        seed=SEED,
        clause_drop_p=args.clause_drop_p,
        batch_size=args.batch_size,
    )

    _LOGGER.info(f"Running {TMClassifier} for {args.epochs}")
    start_time = time()
    for epoch in range(args.epochs):
        benchmark_total = BenchmarkTimer(logger=_LOGGER, text="Epoch Time")
        with benchmark_total:
            benchmark1 = BenchmarkTimer(logger=_LOGGER, text="Training Time")
            with benchmark1:
                tm.fit(X_train, Y_train)
            experiment_results["train_time"].append(benchmark1.elapsed())

            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(X_test) == Y_test).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(
                f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                f"Testing Time: {benchmark2.elapsed():.2f}s"
            )

        if args.device == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
    return experiment_results



def run_mnist(
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
):
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
    _LOGGER.info(f"Running MNIST with {args}")
    experiment_results = metrics(args,  "MNIST")
    data = MNIST().get()

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        seed=SEED,
        clause_drop_p=args.clause_drop_p,
        literal_drop_p=args.literal_drop_p,
        batch_size=args.batch_size,
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

            experiment_results["train_time"].append(benchmark1.elapsed())
            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
    return experiment_results

def run_fashion_mnist(
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
):
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
    _LOGGER.info(f"Running FashionMNIST with {args}")
    experiment_results = metrics(args, "FashionMNIST")
    data = FashionMNIST().get()

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.platform,
        weighted_clauses=args.weighted_clauses,
        seed=SEED,
        clause_drop_p=args.clause_drop_p,
        literal_drop_p=args.literal_drop_p,
        batch_size=args.batch_size,
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

            experiment_results["train_time"].append(benchmark1.elapsed())

            # print(res)
            benchmark2 = BenchmarkTimer(logger=_LOGGER, text="Testing Time")
            with benchmark2:
                result = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()
                experiment_results["accuracy"].append(result)
            experiment_results["test_time"].append(benchmark2.elapsed())

            _LOGGER.info(f"Epoch: {epoch + 1}, Accuracy: {result:.2f}, Training Time: {benchmark1.elapsed():.2f}s, "
                         f"Testing Time: {benchmark2.elapsed():.2f}s")

        if args.platform == "CUDA":
            CudaProfiler().print_timings(benchmark=benchmark_total)

    end_time = time()
    _LOGGER.info(f"Total time taken: {end_time - start_time}")
    experiment_results["total_time"] = end_time - start_time
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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    _LOGGER.addHandler(console_handler)

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

    # run experiments
    # put them all into a tuple
    experiments = [
        (run_mnist, {"num_clauses": 8000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_mnist, {"num_clauses": 6000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_mnist, {"num_clauses": 4000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_fashion_mnist, {"num_clauses": 100, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_fashion_mnist, {"num_clauses": 1000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_fashion_mnist, {"num_clauses": 2000, "T": 6400, "s": 5, "clause_drop_p": 0.25, "epochs": 60}),
        (run_cifar10, {"num_clauses": 10000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60}),
        (run_cifar10, {"num_clauses": 15000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60}),
        (run_cifar10, {"num_clauses": 20000, "T": 48000, "s": 10, "clause_drop_p": 0.5, "epochs": 60}),
    ]

    mnist_ex, fashion_ex, cifar10_ex = experiments[:3], experiments[3:6], experiments[6:]

    for ex in fashion_ex+cifar10_ex:
        func, kwargs = ex
        result = func(**kwargs)
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

