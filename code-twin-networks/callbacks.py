import csv
import os
import time
from enum import Enum
from typing import Dict, List

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
                torch.save([model.kwargs, model.state_dict()], path)
        else:
            print(f"\n'{self.monitor}' did not improve from {self.best}\n")
        return logs


class LRScheduleCallback(Callback):
    def __init__(self, scheduler: ReduceLROnPlateau,
                 monitor: str = None):
        super(LRScheduleCallback, self).__init__()
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs: Dict = None):
        self.scheduler.step(logs[self.monitor])

        return logs
