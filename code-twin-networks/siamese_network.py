from __future__ import print_function

import argparse
import csv
import os
import shutil
import time
from enum import Enum
from typing import List, Callable, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR, \
    CosineAnnealingWarmRestarts
from torch.utils.data import Dataset

from datasets import RandomExclusiveListApply, RandomApply, RandomOrganoidPairDataset, DeterministicOrganoidPairDataset, \
    SquarePad, RandomOrganoidHistPairDataset, DeterministicOrganoidHistPairDataset, OnlineRandomOrganoidHistPairDataset, \
    OnlineDeterministicOrganoidHistPairDataset
from defaults import MODELS_DIR
from model import InputType, SiameseNetwork


class Callback:
    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs: Dict = None):
        pass

    def on_epoch_end(self, epoch, logs: Dict = None) -> Dict:
        pass

    def on_batch_begin(self, batch, logs: Dict = None):
        pass

    def on_batch_end(self, batch, logs: Dict = None):
        pass

    def on_train_begin(self, logs: Dict = None):
        pass

    def on_train_end(self, logs: Dict = None):
        pass


class CallbackContainer:
    def __init__(self, callbacks: List[Callback] = None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            cb_logs = callback.on_epoch_end(epoch, logs)
            if cb_logs:
                logs.update(cb_logs)
        return logs

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = time.time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        # logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        # logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = time.time()
        logs['duration'] = logs['stop_time'] - logs['start_time']
        for callback in self.callbacks:
            callback.on_train_end(logs)


class CSVLogger(Callback):
    def __init__(self, path: str = "summary.csv"):
        super(CSVLogger, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        header_written = os.path.exists(self.path)

        with open(self.path, "a") as f:
            csv_writer = csv.DictWriter(f, fieldnames=list(logs.keys()))
            if not header_written:
                csv_writer.writeheader()

            row_dict = {
                name: (value.item() if isinstance(value, torch.Tensor) else value) for (name, value) in logs.items()
            }
            csv_writer.writerow(rowdict=row_dict)
        return logs


class TimeCallback(Callback):
    def __init__(self):
        super(TimeCallback, self).__init__()
        self.start_time = None

    def on_epoch_begin(self, epoch, logs: Dict = None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs: Dict = None) -> Dict:
        logs = logs or {}
        end_time = time.time()
        duration = end_time - self.start_time
        logs["epoch_duration"] = duration
        return logs


class MonitorMode(Enum):
    MIN = 0
    MAX = 1


class ModelCheckpoint(Callback):
    def __init__(self,
                 models_dict: Dict[str, torch.nn.Module],
                 checkpoint_dir: str,
                 monitor: str = "val_loss",
                 mode: MonitorMode = MonitorMode.MIN
                 ):
        super(Callback, self).__init__()
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.paths_to_models = {
            os.path.join(checkpoint_dir, name): model for name, model in models_dict.items()
        }
        self.monitor = monitor
        self.mode = mode
        self.best = torch.inf if mode == MonitorMode.MIN else -torch.inf

    def on_epoch_end(self, epoch, logs: Dict = None):
        if self.monitor not in logs.keys():
            raise RuntimeWarning(f"Cannot monitor '{self.monitor}' because it is not in the logs")

        if (self.mode == MonitorMode.MIN and (logs[self.monitor] < self.best)) \
                or (self.mode == MonitorMode.MAX and (logs[self.monitor] > self.best)):
            print(f"\n'{self.monitor}' improved from {self.best} to {logs[self.monitor]}.. Saving...\n")
            self.best = logs[self.monitor]

            for path, model in self.paths_to_models.items():
                # # delete checkpoint if exists
                # if os.path.exists(path):
                #     os.remove(path)

                torch.save([model.kwargs, model.state_dict()], path)
        else:
            print(f"\n'{self.monitor}' did not improve from {self.best}\n")
        return logs


class Metric:
    def __init__(self, name: str, precision: int = 2):
        self.name: str = name
        self.precision = precision

    def update(self, predictions, truth):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")

    def reset(self):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")

    def value(self):
        raise NotImplementedError("Calling abstract base method of 'Metric' class")


class MetricContainer:
    def __init__(self, metrics: List[Metric] = None):
        self.metrics: List[Metric] = metrics or []
        self.eval = False

    def set_train(self):
        # reset all metrics
        self.reset()
        self.eval = False

    def set_val(self):
        self.reset()
        self.eval = True

    def update(self, predictions, truth):
        for metric in self.metrics:
            metric.update(predictions, truth)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def summary(self):
        metrics_logs = {}

        for metric in self.metrics:
            name = metric.name
            value = metric.value()

            if isinstance(value, dict):
                for (name, m) in value.items():
                    if self.eval:
                        if not name.startswith("val_"):
                            name = f"val_{name}"
                    metrics_logs[name] = m
            else:
                if self.eval:
                    if not name.startswith("val_"):
                        name = f"val_{name}"
                metrics_logs[name] = value

        return metrics_logs

    def summary_string(self):
        metrics_logs = {}

        for metric in self.metrics:
            name = metric.name
            value = metric.value()
            if isinstance(value, dict):
                for (name, m) in value.items():
                    if self.eval:
                        if not name.startswith("val_"):
                            name = f"val_{name}"
                    metrics_logs[name] = m, metric.precision
            else:
                if self.eval:
                    if not name.startswith("val_"):
                        name = f"val_{name}"
                metrics_logs[name] = value, metric.precision

        return "\t".join(
            [f'{{}}: {{:.{precision}f}}'.format(name, value) for name, (value, precision) in metrics_logs.items()])


class BinaryAccuracy(Metric):
    def __init__(self,
                 threshold=0.5,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super(BinaryAccuracy, self).__init__(name, precision)
        self.threshold = threshold
        self.correct = 0
        self.total = 0

    def update(self, predictions, targets):
        predictions = torch.where(predictions >= self.threshold, 1, 0)
        correct = torch.where(predictions.eq(targets), 1, 0).sum().item()
        self.correct += correct
        self.total += len(predictions)

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return self.correct / self.total


class Accuracy(Metric):
    def __init__(self,
                 embeddings_to_metric: Callable = lambda x: x,
                 pos_min=0.5,
                 pos_max=1.0,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super().__init__(name, precision)
        self.embeddings_to_metric = embeddings_to_metric
        self.correct = 0
        self.total = 0
        self.pos_min = pos_min
        self.pos_max = pos_max

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            predictions = self.embeddings_to_metric(predictions)
        predictions = torch.where(
            torch.logical_and(
                predictions >= self.pos_min, predictions <= self.pos_max
            ),
            1,
            0
        )
        correct = torch.where(predictions.eq(truth), 1, 0).sum().item()
        self.correct += correct
        self.total += len(predictions)

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return self.correct / self.total


class ConfusionMatrix(Metric):
    def __init__(self,
                 embeddings_to_metric: Callable = lambda x: x,
                 pos_min=0.5,
                 pos_max=1.0,
                 precision: int = 2
                 ):
        super().__init__("ConfusionMatrix", precision)
        self.embeddings_to_metric = embeddings_to_metric
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.pos_min = pos_min
        self.pos_max = pos_max

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            predictions = self.embeddings_to_metric(predictions)
        predictions = torch.where(
            torch.logical_and(
                predictions >= self.pos_min, predictions <= self.pos_max
            ),
            1,
            0
        )
        tp = torch.where(torch.logical_and(
            truth == 1,
            predictions == 1
        ), 1, 0).sum().item()
        tn = torch.where(torch.logical_and(
            truth == 0,
            predictions == 0
        ), 1, 0).sum().item()
        fp = torch.where(torch.logical_and(
            truth == 0,
            predictions == 1
        ), 1, 0).sum().item()
        fn = torch.where(torch.logical_and(
            truth == 1,
            predictions == 0
        ), 1, 0).sum().item()

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def value(self):
        return {
            "TP": self.tp,
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn
        }


class CosineConfusionMatrix(ConfusionMatrix):
    def __init__(self,
                 pos_min=0.5,
                 pos_max=1.0,
                 precision: int = 2
                 ):
        super(CosineConfusionMatrix, self).__init__(
            embeddings_to_metric=lambda x: torch.nn.functional.cosine_similarity(x[0], x[1]),
            pos_min=pos_min,
            pos_max=pos_max,
            precision=precision
        )


class ContrastiveConfusionMatrix(ConfusionMatrix):
    def __init__(self,
                 pos_min=0.5,
                 pos_max=1.0,
                 precision: int = 2
                 ):
        super(ContrastiveConfusionMatrix, self).__init__(
            embeddings_to_metric=lambda x: torch.nn.functional.pairwise_distance(x[0], x[1]),
            pos_min=pos_min,
            pos_max=pos_max,
            precision=precision
        )


class ContrastiveAccuracy(Accuracy):
    def __init__(self,
                 pos_min=0.5,
                 pos_max=1.0,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super(ContrastiveAccuracy, self).__init__(
            embeddings_to_metric=lambda x: torch.nn.functional.pairwise_distance(x[0], x[1]),
            pos_min=pos_min,
            pos_max=pos_max,
            name=name,
            precision=precision
        )


class CosineAccuracy(Accuracy):
    def __init__(self,
                 pos_min=0.5,
                 pos_max=1.0,
                 name: str = "accuracy",
                 precision: int = 2
                 ):
        super(CosineAccuracy, self).__init__(
            embeddings_to_metric=lambda x: torch.nn.functional.cosine_similarity(x[0], x[1]),
            pos_min=pos_min,
            pos_max=pos_max,
            name=name,
            precision=precision
        )


class AverageDistanceMetric(Metric):
    def __init__(self, target, name: str = "distance", precision: int = 2):
        super().__init__(name, precision)
        self.target = target
        self.total_distance = 0
        self.count = 0

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            out1, out2 = predictions
            # calculate the distance
            predictions = torch.nn.functional.pairwise_distance(out1, out2)

        target_distances = torch.where(
            torch.eq(truth, self.target), predictions, torch.scalar_tensor(0.).cuda()
        )
        self.total_distance += torch.sum(target_distances)
        target_num = torch.sum(torch.where(torch.eq(truth, self.target), 1, 0).sum())
        self.count += target_num

    def reset(self):
        self.total_distance = 0
        self.count = 0

    def value(self):
        return self.total_distance / self.count


class AverageCosineMetric(Metric):
    def __init__(self, target, name: str = "distance", precision: int = 2):
        super().__init__(name, precision)
        self.target = target
        self.total_distance = 0
        self.count = 0

    def update(self, predictions, truth):
        if isinstance(predictions, tuple):
            out1, out2 = predictions
            # calculate the distance
            predictions = torch.nn.functional.cosine_similarity(out1, out2)

        target_distances = torch.where(
            torch.eq(truth, self.target), predictions, torch.scalar_tensor(0.).cuda()
        )
        self.total_distance += torch.sum(target_distances)
        target_num = torch.sum(torch.where(torch.eq(truth, self.target), 1, 0).sum())
        self.count += target_num

    def reset(self):
        self.total_distance = 0
        self.count = 0

    def value(self):
        return self.total_distance / self.count


class CosineEmbeddingLossWrapper(torch.nn.Module):
    def __init__(self, margin: float = 0.0, *args, **kwargs):
        super(CosineEmbeddingLossWrapper, self).__init__()
        self.cosine_embedding_loss = torch.nn.CosineEmbeddingLoss(margin=margin, *args, **kwargs)

    def forward(self, inputs: tuple, target: torch.Tensor) -> torch.Tensor:
        input0, input1 = inputs
        target = torch.where(target == 1, 1, -1)
        return self.cosine_embedding_loss.forward(input0, input1, target)


class LRScheduleCallback(Callback):
    def __init__(self, scheduler: Union[StepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingLR],
                 monitor: str = None):
        super(LRScheduleCallback, self).__init__()
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs: Dict = None):
        if isinstance(self.scheduler, StepLR) or isinstance(self.scheduler, ExponentialLR) or isinstance(self.scheduler,
                                                                                                         CosineAnnealingLR) or isinstance(
            self.scheduler, CosineAnnealingWarmRestarts):
            self.scheduler.step()
            print(self.scheduler.get_last_lr())
        elif isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(logs[self.monitor])

        return logs


class EarlyStoppingCallback(Callback):
    def __init__(self,
                 monitor: str = "val_loss",
                 mode: MonitorMode = MonitorMode.MIN,
                 patience: int = 10
                 ):
        super(EarlyStoppingCallback, self).__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.count = 0
        self.current_best = torch.inf if mode == MonitorMode.MIN else -torch.inf

    def on_epoch_end(self, epoch, logs: Dict = None):
        logs = logs or {}

        if (self.mode == MonitorMode.MIN and self.current_best > logs[self.monitor]) \
                or (self.mode == MonitorMode.MAX and self.current_best < logs[self.monitor]):
            # improved, reset counter and current best
            print(f"Early Stopper: '{self.monitor}' improved from {self.current_best} to {logs[self.monitor]}\n")
            self.count = 0
            self.current_best = logs[self.monitor]
        else:
            # did not improve
            self.count += 1
            if self.count >= self.patience:
                # somehow flag that training should be stopped
                print("Flagging for train termination...")
                logs.update({"stop_training": True})
        return logs


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 callbacks: CallbackContainer = None,
                 metrics: MetricContainer = None,
                 log_interval=10,
                 dry_run=False,
                 output_regularizer=0.0
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or CallbackContainer()
        self.metrics = metrics or MetricContainer()
        self.log_interval = log_interval
        self.dry_run = dry_run
        self.output_regularizer = output_regularizer

    def train(self,
              epochs: int,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader = None
              ):

        for epoch in range(1, epochs + 1):
            logs = {

            }
            self.callbacks.on_epoch_begin(epoch, logs)
            train_logs = self.train_epoch(epoch, train_loader)
            val_logs = self.val_epoch(epoch, val_loader)
            logs.update(train_logs)
            logs.update(val_logs)
            logs = self.callbacks.on_epoch_end(epoch, logs)
            if "stop_training" in logs.keys() and logs["stop_training"]:
                break

    def train_epoch(self, epoch, train_loader: torch.utils.data.DataLoader):
        self.model.train()
        self.metrics.set_train()

        logs = {
            "epoch": epoch,
            "loss": 0,
        }

        batch_idx = 0
        for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
            images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images_1, images_2)

            self.metrics.update(outputs, targets)
            loss = self.criterion(outputs, targets)
            logs["loss"] += loss.item()

            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tBCELoss: {:.6f}\t RegLoss: {:.6f}\t {}'.format(
                    epoch, batch_idx * len(images_1), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader),
                           logs["loss"] / (batch_idx + 1), self.metrics.summary_string()
                      ))
                if self.dry_run:
                    break
        logs["loss"] /= (batch_idx + 1)

        logs.update(self.metrics.summary())
        self.metrics.reset()
        return logs

    def val_epoch(self, epoch, val_loader: torch.utils.data.DataLoader) -> Dict:
        self.model.eval()
        self.metrics.set_val()

        logs = {
            "val_loss": 0,
            "epoch": epoch
        }

        with torch.no_grad():
            for batch_idx, (images_1, images_2, targets) in enumerate(val_loader):
                images_1, images_2, targets = images_1.to(self.device), images_2.to(self.device), targets.to(
                    self.device)
                if isinstance(self.model, SiameseNetwork) and self.model.return_penultimate:
                    outputs, penultimate = self.model(images_1, images_2)
                elif isinstance(self.model, SiameseNetwork) and not self.model.return_penultimate:
                    outputs = self.model(images_1, images_2)
                else:
                    raise ValueError("Network must be SiameseNetwork.")

                logs["val_loss"] += self.criterion(outputs, targets).item()  # sum up batch loss

                # similarities = nn.functional.pairwise_distance(outputs1, outputs2)
                self.metrics.update(outputs, targets)
                # pred = torch.where(similarities < 2.0, 1, 0)  # get the index of the max log-probability
                # correct_test += pred.eq(targets.view_as(pred)).sum().item()
                if batch_idx % self.log_interval == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\t'.format(
                        epoch, batch_idx * len(images_1), len(val_loader.dataset),
                               100. * batch_idx / len(val_loader)))
                if self.dry_run:
                    break

        logs["val_loss"] /= (batch_idx + 1)

        print('\nTest set: Average BCE loss: {:.4f}\t Average Reg. loss: {:.4f}\t'.format(logs["val_bce_loss"], logs[
            "val_reg_loss"]) + self.metrics.summary_string() + "\n")

        logs.update(self.metrics.summary())
        self.metrics.reset()

        return logs


def get_metrics() -> MetricContainer:
    return MetricContainer([
        BinaryAccuracy(name='accuracy', precision=3),
        ConfusionMatrix(pos_min=0.5, pos_max=1.0, precision=0)
    ])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Siamese network for Organoids')
    parser.add_argument('--data-dirs',
                        nargs='+',
                        default=["/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/split/"],
                        help="Data directories for input files"
                        )
    parser.add_argument('--val-data-dirs',
                        nargs='+',
                        default=None,
                        help="Optional validation data directory"
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size',
                        type=int,
                        default=128,
                        metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma',
                        type=float,
                        default=0.7,
                        metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0,
                        help='Weight decay parameter for the optimizer (default: 0.0)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name',
                        type=str,
                        help='model name',
                        default="p0.2-256x256-squarePad-color-jitter-correct"
                        )
    parser.add_argument('--override',
                        help='Whether to override models under the same name',
                        action='store_true',
                        default=False
                        )
    parser.add_argument('--margin',
                        type=float,
                        default=None,
                        help="Margin for either contrastive loss or cosine loss"
                        )
    parser.add_argument('--input-type',
                        type=InputType,
                        default=InputType.IMAGES,
                        choices=[ip for ip in InputType]
                        )
    parser.add_argument('--embedding-dimension',
                        type=int,
                        default=128
                        )
    parser.add_argument('--model-dir',
                        type=str,
                        default="./image-models-colorjitter-2",
                        help="Where to save checkpoints and such."
                        )
    parser.add_argument('--total-steps',
                        type=int,
                        default=6000,
                        help='How many steps to train for in total'
                        )
    parser.add_argument('--steps-per-epoch',
                        type=int,
                        default=100,
                        help='How many batches to train for per epoch'
                        )
    parser.add_argument("--output-regularizer",
                        type=float,
                        default=0.0,
                        help="Regularization directly for the output."
                        )
    args = parser.parse_args()

    if args.val_data_dirs is None:
        args.val_data_dirs = args.data_dirs

    epochs = args.total_steps // args.steps_per_epoch

    model_dir = os.path.join(args.model_dir, args.model_name)
    if args.override and os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "siamese_network.pt")

    if os.path.exists(model_path) and not args.override:
        print("Model exists. Specify --override to override")
        exit()
    if os.path.exists(model_path) and args.override:
        shutil.rmtree(model_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        print("Using CUDA device...")
        device = torch.device("cuda")
    else:
        print("Using CPU...")
        device = torch.device("cpu")

    train_kwargs = {
        'batch_size': args.batch_size
    }
    test_kwargs = {
        'batch_size': args.val_batch_size
    }
    if use_cuda:
        num_workers = len(os.sched_getaffinity(0))
        print(f"Running with {num_workers} workers...")
        cuda_kwargs = {
            'num_workers': num_workers,
            'pin_memory': True,
            'shuffle': True
        }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.input_type == InputType.IMAGES:
        train_dataset = RandomOrganoidPairDataset(
            args.data_dirs,
            split="train",
            num_batches=args.steps_per_epoch,
            batch_size=args.batch_size,
            transforms=torchvision.transforms.Compose(
                [
                    SquarePad(),
                    torchvision.transforms.Resize(
                        (256, 256),
                        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                        antialias=True
                    ),
                    torchvision.transforms.RandomHorizontalFlip(p=0.2),
                    torchvision.transforms.RandomVerticalFlip(p=0.2),
                    RandomApply(RandomExclusiveListApply(
                        choice_modules=nn.ModuleList([
                            torchvision.transforms.RandomRotation(degrees=180)
                        ])
                    ), p=0.2)
                ]
            ))
        # no transforms here
        test_dataset = DeterministicOrganoidPairDataset(
            args.val_data_dirs,
            split="test-seen",
            transforms=torch.nn.Sequential(
                SquarePad(),
                torchvision.transforms.Resize(
                    (256, 256),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ))
    elif args.input_type == InputType.HISTOGRAM:
        train_dataset = OnlineRandomOrganoidHistPairDataset(
            args.data_dirs,
            split="train",
            num_batches=args.steps_per_epoch,
            batch_size=args.batch_size,
            transforms=None
        )
        # no transforms here
        test_dataset = OnlineDeterministicOrganoidHistPairDataset(
            args.val_data_dirs,
            split="test-seen"
        )
    else:
        raise NotImplementedError(f"Input Type {args.input_type} not recognized.")

    print("Train Set Size: ", len(train_dataset))
    print("Val Set Size: ", len(test_dataset))
    print(
        f"Training on {len(train_dataset.organoid_classes)} organoid classes. Validating on {len(set(test_dataset.organoid_classes))} organoid classes.")

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # model building phase
    model = SiameseNetwork(
        input_type=args.input_type,
        embedding_dimension=args.embedding_dimension,
        return_penultimate=args.output_regularizer != 0.0
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics = get_metrics()

    trainer = Trainer(
        model=model,
        log_interval=10,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        metrics=metrics,
        output_regularizer=args.output_regularizer,
        callbacks=CallbackContainer([
            TimeCallback(),
            CSVLogger(os.path.join(model_dir, "history.csv")),
            EarlyStoppingCallback(monitor="val_accuracy", patience=300, mode=MonitorMode.MAX),
            ModelCheckpoint(
                monitor="val_bce_loss",
                models_dict={
                    "siamese.pt": model
                },
                checkpoint_dir=os.path.join(model_dir, "val_loss_checkpoints")
            ),
            ModelCheckpoint(
                monitor='val_accuracy',
                mode=MonitorMode.MAX,
                models_dict={
                    "siamese.pt": model
                },
                checkpoint_dir=os.path.join(model_dir, "val_accuracy_checkpoints")
            ),
            LRScheduleCallback(
                ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=20
                ),
                monitor="bce_loss"
            )
        ])
    )

    trainer.train(epochs, train_loader, val_loader)

    if not os.path.exists(model_path):
        torch.save([model.kwargs, model.state_dict()], model_path)


if __name__ == '__main__':
    main()
