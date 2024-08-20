import argparse
import os
from enum import Enum

import h5py
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import DeterministicOrganoidDataset, DeterministicOrganoidHistDataset
from siamese_network import SiameseNetwork, SquarePad, InputType


class AnalysisMode(Enum):
    LATEST_MODEL = 0
    BEST_VAL_ACCURACY = 1
    BEST_VAL_LOSS = 2


ANALYSIS_SUBDIR = "analysis"


def load_siamese_network(
        model_dir: str,
        analysis_mode: AnalysisMode,
        device: str = "cpu"
):
    if analysis_mode == AnalysisMode.LATEST_MODEL:
        model_path = os.path.join(model_dir, "siamese_network.pt")
    elif analysis_mode == AnalysisMode.BEST_VAL_LOSS:
        model_path = os.path.join(model_dir, "val_loss_checkpoints", "siamese.pt")
    elif analysis_mode == AnalysisMode.BEST_VAL_ACCURACY:
        model_path = os.path.join(model_dir, "val_accuracy_checkpoints", "siamese.pt")
    else:
        raise ValueError(f"Analysis Mode {analysis_mode} not supported")

    kwargs, model_state_dict = torch.load(model_path, map_location=device)

    model = SiameseNetwork(**kwargs)

    model.load_state_dict(model_state_dict)
    model = model.to(device)

    return model


class TwoInputSequential(torch.nn.Module):
    def __init__(self, layers):
        super(TwoInputSequential, self).__init__()
        self.layers = layers

    def forward(self, input1, input2):
        out = self.layers[0](input1, input2)
        for layer in self.layers[1:]:
            out = layer(out)
        return out


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels):
        super(PairwiseDataset, self).__init__()
        self.inputs = inputs
        self.labels = labels

        assert (len(inputs) == len(labels))

    def __len__(self):
        return len(self.inputs) ** 2

    def __getitem__(self, index):
        index1 = index % len(self.inputs)
        index2 = (index - index1) // len(self.inputs)

        return (index1, self.inputs[index1], self.labels[index1]), (index2, self.inputs[index2], self.labels[index2])


def get_subdir(analysis_mode: AnalysisMode):
    if analysis_mode == AnalysisMode.LATEST_MODEL:
        return ""
    elif analysis_mode == AnalysisMode.BEST_VAL_LOSS:
        return "val_loss_checkpoints"
    elif analysis_mode == AnalysisMode.BEST_VAL_ACCURACY:
        return "val_accuracy_checkpoints"


def analyse_test_set_only(
        model_dir: str,
        data_dir: str,
        analysis_mode: AnalysisMode = AnalysisMode.LATEST_MODEL,
        out_fn: str = None,
        subset_name: str = 'TEST',
        input_type: InputType = InputType.IMAGES
):
    if out_fn is None:
        raise ValueError("'out_fn' cannot be None.")

    out_dir = os.path.join(model_dir, get_subdir(analysis_mode), ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    siamese_network = load_siamese_network(model_dir, analysis_mode, device=device)

    siamese_network.eval()

    if input_type == InputType.IMAGES:
        dataset = DeterministicOrganoidDataset(
            [data_dir],
            transforms=torch.nn.Sequential(
                SquarePad(),
                torchvision.transforms.Resize(
                    (256, 256),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ),
            return_file_ids=True
        )
    elif input_type == InputType.HISTOGRAM:
        dataset = DeterministicOrganoidHistDataset(
            [data_dir],
            return_file_ids=True
        )
    else:
        raise ValueError(f"InputType {input_type} not known.")

    print("Dataset Length: ", len(dataset))

    dataloader = DataLoader(dataset, batch_size=32, num_workers=os.cpu_count())

    affinities = np.zeros((len(dataset), len(dataset)), dtype=np.float32)
    labels = []
    file_ids = []

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to(device)

        for batch_idx, (images, current_labels, current_file_ids) in enumerate(tqdm(dataloader)):
            images = images.to(device)
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            labels.extend(current_labels)
            file_ids.extend([os.path.split(fid)[1] for fid in current_file_ids])

        pair_ds = PairwiseDataset(embeddings.cpu(), labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(tqdm(pair_ds_loader)):
            images1, images2 = images1.to(device), images2.to(device)
            siamese_output = classifier_head(images1, images2)

            for num in range(len(siamese_output)):
                index1 = int(indices1[num])
                index2 = int(indices2[num])

                if index1 != index2:
                    affinities[index1, index2] = float(siamese_output[num])

    with h5py.File(out_path, 'w') as hdf_f:
        hdf_f.create_dataset('affinities', data=affinities, dtype='float32')
        hdf_f.create_dataset('labels', data=labels, dtype=h5py.string_dtype('utf-8'))
        hdf_f.create_dataset('subsets', data=[subset_name] * len(labels), dtype=h5py.string_dtype('utf-8'))
        hdf_f.create_dataset('file_ids', data=file_ids, dtype=h5py.string_dtype('utf-8'))


def analyse_unseen_and_test_set(
        model_dir: str,
        unseen_data_dir: str,
        test_set_data_dir: str,
        out_fn: str,
        analysis_mode: AnalysisMode = AnalysisMode.LATEST_MODEL,
        input_type: InputType = InputType.IMAGES
):
    out_dir = os.path.join(model_dir, get_subdir(analysis_mode), ANALYSIS_SUBDIR)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path = os.path.join(out_dir, out_fn)

    if os.path.isfile(out_path):
        print("All-Pairs-Matrix already exists.. Skipping..")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    siamese_network = load_siamese_network(model_dir, analysis_mode, device=device)

    siamese_network.eval()

    if input_type == InputType.IMAGES:
        unseen_dataset = DeterministicOrganoidDataset(
            [unseen_data_dir],
            transforms=torch.nn.Sequential(
                SquarePad(),
                torchvision.transforms.Resize(
                    (256, 256),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ),
            return_file_ids=True
        )
        test_data_set = DeterministicOrganoidDataset(
            [test_set_data_dir],
            transforms=torch.nn.Sequential(
                SquarePad(),
                torchvision.transforms.Resize(
                    (256, 256),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True
                )
            ),
            return_file_ids=True
        )
    elif input_type == InputType.HISTOGRAM:
        unseen_dataset = DeterministicOrganoidHistDataset(
            [unseen_data_dir],
            return_file_ids=True
        )
        test_data_set = DeterministicOrganoidHistDataset(
            [test_set_data_dir],
            return_file_ids=True
        )
    else:
        raise ValueError(f"InputType {input_type} not known.")

    num_elements = len(unseen_dataset) + len(test_data_set)

    affinities = np.zeros((num_elements, num_elements), dtype=np.float32)
    labels = []
    file_ids = []
    subsets = []

    print(f"Unseen Set Size {len(unseen_dataset)} organoid samples...")
    print(f"Test Set Size {len(test_data_set)} organoid samples...")

    test_set_dataloader = DataLoader(test_data_set, batch_size=1024, num_workers=os.cpu_count())

    dataloader = DataLoader(unseen_dataset, batch_size=1024, num_workers=os.cpu_count())

    with torch.no_grad():

        embedding_network = list(siamese_network.children())[0]
        classifier_head = TwoInputSequential(list(siamese_network.children())[1:])

        embeddings = torch.empty(size=(0,)).to('cuda')

        for batch_idx, (images, current_labels, current_file_ids) in enumerate(tqdm(dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            labels.extend(current_labels)
            file_ids.extend([os.path.split(fid)[1] for fid in current_file_ids])

        for batch_idx, (images, current_labels, current_file_ids) in enumerate(tqdm(test_set_dataloader)):
            images = images.to('cuda')
            embeddings = torch.cat([embeddings, embedding_network(images)], dim=0)
            labels.extend(current_labels)
            file_ids.extend([os.path.split(fid)[1] for fid in current_file_ids])

        subsets.extend(['UNSEEN'] * len(unseen_dataset))
        subsets.extend(['TEST'] * len(test_data_set))

        pair_ds = PairwiseDataset(embeddings.cpu(), labels)
        pair_ds_loader = DataLoader(pair_ds, batch_size=2 ** 18, num_workers=os.cpu_count())

        for batch_idx, ((indices1, images1, labels1), (indices2, images2, labels2)) in enumerate(
                tqdm(pair_ds_loader)):
            images1, images2 = images1.to('cuda'), images2.to('cuda')
            siamese_output = classifier_head(images1, images2)

            for num in range(len(siamese_output)):
                index1 = int(indices1[num])
                index2 = int(indices2[num])

                if index1 != index2:
                    affinities[index1, index2] = float(siamese_output[num])

    with h5py.File(out_path, 'w') as hdf_f:
        hdf_f.create_dataset('affinities', data=affinities, dtype='float32')
        hdf_f.create_dataset('labels', data=labels, dtype=h5py.string_dtype('utf-8'))
        hdf_f.create_dataset('subsets', data=subsets, dtype=h5py.string_dtype('utf-8'))
        hdf_f.create_dataset('file_ids', data=file_ids, dtype=h5py.string_dtype('utf-8'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test-100-dir',
        type=str,
        default='../data/test-100'
    )
    parser.add_argument(
        '--test-30-dir',
        type=str,
        default='../data/test-30'
    )
    parser.add_argument('--input-type',
                        type=InputType,
                        default=InputType.IMAGES,
                        choices=[ip for ip in InputType]
                        )
    parser.add_argument('--model-dir',
                        type=str,
                        default="./models/tni-p0.2",
                        required=False)
    args = parser.parse_args()

    analyse_test_set_only(
        model_dir=args.model_dir,
        data_dir=args.test_100_dir,
        out_fn=f"test.h5",
        analysis_mode=AnalysisMode.LATEST_MODEL,
        input_type=args.input_type,
    )

    analyse_test_set_only(
        model_dir=args.model_dir,
        data_dir=args.test_30_dir,
        out_fn=f"test-unseen.h5",
        analysis_mode=AnalysisMode.LATEST_MODEL,
        input_type=args.input_type,
    )

    analyse_unseen_and_test_set(
        model_dir=args.model_dir,
        test_set_data_dir=args.test_100_dir,
        unseen_data_dir=args.test_30_dir,
        out_fn="test-and-unseen.h5",
        analysis_mode=AnalysisMode.LATEST_MODEL,
        input_type=args.input_type
    )