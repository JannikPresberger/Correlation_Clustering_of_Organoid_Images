import csv
import os
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Sequential, Conv2d, Linear, BatchNorm2d, Conv1d, BatchNorm1d
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Concatenation
from siamese_network import SiameseNetwork, SquarePad, InputType
import torch, torchvision
import h5py
from datasets import DeterministicOrganoidDataset, DeterministicOrganoidHistDataset


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



def plot_roc_curves(in_dir: str):


    for file in Path(in_dir).glob("**/*_roc.csv"):
        out_path = str(file).replace(".csv", ".png")
        out_path_join_thresholds = str(file).replace(".csv", "_join_thresholds.png")

        out_path_cut_thresholds = str(file).replace(".csv", "_cut_thresholds.png")

        with open(file, "r") as f:
            reader = csv.DictReader(f)

            data = [line for line in reader]

        for line in data:
            line["threshold"] = float(line["threshold"])
            line["tj"] = float(line["tj"])
            line["tc"] = float(line["tc"])
            line["fc"] = float(line["fc"])
            line["fj"] = float(line["fj"])

        thresholds = [line["threshold"] for line in data]

        cut_precisions = [
            line["tc"]/(line["tc"] + line["fc"]) if (line["tc"] + line["fc"] > 0) else 1 for line in data
        ]
        cut_recall = [
            line["tc"]/(line["tc"] + line["fj"]) for line in data
        ]
        join_precision = [
            line["tj"]/(line["tj"] + line["fj"]) if (line["tj"] + line["fj"] > 0) else 1 for line in data
        ]
        join_recall = [
            line["tj"]/(line["tj"] + line["fc"]) for line in data
        ]


        plt.figure()
        plt.title("PR-Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(cut_recall, cut_precisions, label="Cuts", marker=".")
        plt.plot(join_recall, join_precision, label="Joins", marker=".")
        plt.legend()
        plt.savefig(out_path)
        plt.close()

        plt.figure()
        plt.title("Joins")
        plt.xlabel("Threshold")
        plt.plot(thresholds, join_recall, label="Recall", marker=".")
        plt.plot(thresholds, join_precision, label="Precision", marker=".")
        plt.legend()
        plt.savefig(out_path_join_thresholds)
        plt.close()

        plt.figure()
        plt.title("Cuts")
        plt.xlabel("Threshold")
        plt.plot(thresholds, cut_recall, label="Recall", marker=".")
        plt.plot(thresholds, cut_precisions, label="Precision", marker=".")
        plt.legend()
        plt.savefig(out_path_cut_thresholds)
        plt.close()

def plot_cost_distributions(in_dir: str):


    for file in Path(in_dir).glob("**/*.h5"):
        true_joins = []
        true_cuts = []

        with h5py.File(file) as f:
            labels = f["labels"][()].astype(dtype=int)
            affinities = f["affinities"][()]

            for i in range(len(labels)):
                for j in range(i):
                    if labels[i] == labels[j]:
                        true_joins.append(1 - (affinities[i, j] + affinities[j, i]) / 2)
                    else:
                        true_cuts.append(1 - (affinities[i, j] + affinities[j, i]) / 2)

            true_joins = np.array(true_joins)
            true_cuts = np.array(true_cuts)

            true_joins = np.sort(true_joins)
            yj = np.cumsum(true_joins)
            yj = yj / yj[-1]
            true_cuts = np.sort(true_cuts)
            yc = np.cumsum(true_cuts)
            yc = yc / yc[-1]

            plt.figure()
            plt.xlim(-0.1, 1.1)
            plt.ylim(0, 1)
            plt.xlabel("p (cut)")
            plt.ylabel("Fraction of Coefficients")
            plt.plot(true_joins, yj, label="True Joins")
            plt.plot(true_cuts, yc, label="True Cuts")
            plt.legend()
            plt.savefig(str(file).replace(".h5", "_cum-distributions.png"))
            plt.close()



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

        subsets.extend(['UNSEEN']*len(unseen_dataset))
        subsets.extend(['TEST']*len(test_data_set))

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


def analyze_model_weights(
        model_dir: str,
        analysis_mode: AnalysisMode = AnalysisMode.LATEST_MODEL
):
    if analysis_mode == AnalysisMode.LATEST_MODEL:
        out_dir = os.path.join(model_dir, "analysis")
    elif analysis_mode == AnalysisMode.BEST_VAL_LOSS:
        out_dir = os.path.join(model_dir, "val_loss_checkpoints", "analysis")
    elif analysis_mode == AnalysisMode.BEST_VAL_ACCURACY:
        out_dir = os.path.join(model_dir, "val_accuracy_checkpoints", "analysis")
    else:
        raise ValueError("Unknown Analysis Mode.")

    out_dir = os.path.join(out_dir, "model_weights")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    def get_params_(model):
        return filter(lambda p: p.requires_grad, model.parameters())
    def get_param_number(model):
        model_parameters = get_params_(model)
        return sum([np.prod(p.size()) for p in model_parameters])

    model = load_siamese_network(model_dir, analysis_mode)

    model.eval()


    # print(model)
    # print("Trainable Model Parameters: ", get_param_number(model))

    minimum = +10000000000
    maximum = -10000000000

    params_number = 0
    for name, module in model.named_modules():
        # print(name, module)
        if isinstance(module, (Conv2d, Conv1d, BatchNorm2d, BatchNorm1d, Linear)):
            params_number += get_param_number(module)

            params = list(get_params_(module))

            count = 0
            for param_group in params:
                minimum = min(minimum, np.min(param_group.detach().numpy().flatten()))
                maximum = max(maximum, np.max(param_group.detach().numpy().flatten()))

                plt.figure()
                plt.hist(param_group.detach().numpy().flatten(), bins=100)

                plt.savefig(os.path.join(out_dir, f"{type(module).__name__}_{name}_{count}.pdf"))
                plt.close()
                count += 1

    print("Minimum Parameter Value: ", minimum)
    print("Maximum Parameter Value: ", maximum)


def check_symmetry(
        path_to_h5_file: str
):
    with h5py.File(path_to_h5_file, "r") as f:
        affinities = np.array(f["affinities"])
        log_likelihoods = np.log((1-affinities)/affinities)
        symmetric_difference = affinities - affinities.T

        print("Min Difference: ", symmetric_difference.min())
        print("Max Difference: ", symmetric_difference.max())

        plt.figure()
        plt.hist(symmetric_difference.flatten(), bins=300)
        # plt.show()

        too_big = (affinities > 0.99).sum()
        # correct for diagonal elements
        too_low = (affinities < 0.01).sum() - affinities.shape[0]

        print("Too Big: ", too_big / (affinities.size - affinities.shape[0]))
        print("Too Small", too_low / (affinities.size - affinities.shape[0]))

        print("Minimum Log-Odds: ", log_likelihoods.min())
        print("Maximum Log-Odds: ", log_likelihoods.max())

        too_big_log = (log_likelihoods > 4).sum()
        too_low_log = (log_likelihoods < -4).sum() - affinities.shape[0]

        print("Too Big Log-Odds: ", too_big_log / (affinities.size - affinities.shape[0]))
        print("Too Small Log-Odds:", too_low_log / (affinities.size - affinities.shape[0]))




if __name__ == "__main__":
    # plot_cost_distributions("./models")
    # plot_roc_curves("./models")


    # check_symmetry("/home/dstein/GitRepos/PhD/organoid-matching/new-image-models/p0.0-256x256-squarePad/analysis/test.h5")


    mdir = "./histogram-models-colorjitter"

    # for model_dir in os.listdir(mdir):
    #     analyze_model_weights(
    #         model_dir=os.path.join(mdir, model_dir),
    #         analysis_mode=AnalysisMode.LATEST_MODEL
    #     )

    # for val_set in validation_sets:
    for model_dir in os.listdir(mdir):

        # print(f"Evaluation for {model_dir}")
        # print(100*"=")
        # print("test.h5")
        # print(100*"-")
        #
        # if os.path.exists(os.path.join(mdir, model_dir, "analysis", "test.h5")):
        #     check_symmetry(os.path.join(mdir, model_dir, "analysis", "test.h5"))
        # else:
        #     print("Doesn't exist.")
        # print("test-and-unseen.h5")
        # print(100*"-")
        # if os.path.exists(os.path.join(mdir, model_dir, "analysis", "test-and-unseen.h5")):
        #     check_symmetry(os.path.join(mdir, model_dir, "analysis", "test-and-unseen.h5"))
        # else:
        #     print("Doesnt exist.")
        #
        # print("unseen.h5")
        # print(100*"-")
        # if os.path.exists(os.path.join(mdir, model_dir, "analysis", "test-unseen.h5")):
        #     check_symmetry(os.path.join(mdir, model_dir, "analysis", "test-unseen.h5"))
        # else:
        #     print("Doesn't exist.")
        #
        # print()


        analyse_test_set_only(
            model_dir=os.path.join(mdir, model_dir),
            data_dir=f"/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/histograms/test",
            out_fn=f"test.h5",
            analysis_mode=AnalysisMode.LATEST_MODEL,
            input_type=InputType.HISTOGRAM if ("hist" in mdir) else InputType.IMAGES
        )

        analyse_test_set_only(
            model_dir=os.path.join(mdir, model_dir),
            data_dir=f"/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/histograms/test-unseen",
            out_fn=f"test-unseen.h5",
            analysis_mode=AnalysisMode.LATEST_MODEL,
            input_type=InputType.HISTOGRAM if ("hist" in mdir) else InputType.IMAGES
        )

        analyse_unseen_and_test_set(
            model_dir=os.path.join(mdir, model_dir),
            test_set_data_dir=f"/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/histograms/test",
            unseen_data_dir=f"/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/histograms/test-unseen",
            out_fn="test-and-unseen.h5",
            analysis_mode=AnalysisMode.LATEST_MODEL,
            input_type=InputType.HISTOGRAM if ("hist" in mdir) else InputType.IMAGES
        )
