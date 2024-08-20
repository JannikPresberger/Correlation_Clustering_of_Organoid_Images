from __future__ import print_function

import argparse
import os
import shutil
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset

from callbacks import CallbackContainer, LRScheduleCallback, TimeCallback, CSVLogger, ModelCheckpoint, MonitorMode
from datasets import RandomExclusiveListApply, RandomApply, RandomOrganoidPairDataset, DeterministicOrganoidPairDataset, \
    SquarePad, OnlineRandomOrganoidHistPairDataset, OnlineDeterministicOrganoidHistPairDataset
from metrics import BinaryAccuracy, MetricContainer, ConfusionMatrix
from model import InputType, SiameseNetwork


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 callbacks: CallbackContainer = None,
                 metrics: MetricContainer = None,
                 log_interval=10,
                 dry_run=False
                 ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.callbacks = callbacks or CallbackContainer()
        self.metrics = metrics or MetricContainer()
        self.log_interval = log_interval
        self.dry_run = dry_run

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
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t {}'.format(
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
                outputs = self.model(images_1, images_2)

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

        print('\nTest set: Average Loss: {:.4f}\t {}'.format(logs["val_loss"], self.metrics.summary_string() + "\n"))

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
    parser.add_argument('--data-dir',
                        type=str,
                        default="../data/train-100",
                        help="Data directories for input files"
                        )
    parser.add_argument('--val-data-dir',
                        type=str,
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
                        help='learning rate (default: 0.0001)')
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
                        default="test"
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
                        default="./twin-network-image-models",
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
    parser.add_argument('--augment',
                        type=float,
                        help="Augmentation Probability",
                        default=0.0
                        )
    args = parser.parse_args()

    if args.val_data_dir is None:
        args.val_data_dir = args.data_dir

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
            args.data_dir,
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
                    torchvision.transforms.RandomHorizontalFlip(p=args.augment),
                    torchvision.transforms.RandomVerticalFlip(p=args.augment),
                    RandomApply(RandomExclusiveListApply(
                        choice_modules=nn.ModuleList([
                            torchvision.transforms.RandomRotation(degrees=180)
                        ])
                    ), p=args.augment)
                ]
            ))
        # no transforms here
        test_dataset = DeterministicOrganoidPairDataset(
            args.val_data_dir,
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
            args.data_dir,
            num_batches=args.steps_per_epoch,
            batch_size=args.batch_size
        )
        test_dataset = OnlineDeterministicOrganoidHistPairDataset(
            args.val_data_dir
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
        embedding_dimension=args.embedding_dimension
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    metrics = get_metrics()

    trainer = Trainer(
        model=model,
        log_interval=10,
        optimizer=optimizer,
        criterion=nn.BCELoss(),
        device=device,
        metrics=metrics,
        callbacks=CallbackContainer([
            TimeCallback(),
            CSVLogger(os.path.join(model_dir, "history.csv")),
            ModelCheckpoint(
                monitor="val_loss",
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
                monitor="loss"
            )
        ])
    )

    trainer.train(epochs, train_loader, val_loader)

    if not os.path.exists(model_path):
        torch.save([model.kwargs, model.state_dict()], model_path)


if __name__ == '__main__':
    main()
