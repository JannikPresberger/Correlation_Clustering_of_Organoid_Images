import csv
import os
import shutil
from collections import defaultdict


def adjust_directory_structure(
        in_dir: str,
        images_per_class,
        out_dir: str
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for file in os.listdir(in_dir):
        if file.endswith("_mask.tif"):
            continue

        file_index = int(os.path.splitext(file)[0].split("_")[-1])

        sub_dir = os.path.join(out_dir, str(file_index // images_per_class))

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        shutil.copy2(os.path.join(in_dir, file), os.path.join(sub_dir, file))


def train_test_test_split(
        in_dir: str,
        out_dir: str
):
    train_dir = os.path.join(out_dir, "train")
    test_dir = os.path.join(out_dir, "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    for directory in os.listdir(in_dir):

        if not os.path.exists(os.path.join(train_dir, directory)):
            os.makedirs(os.path.join(train_dir, directory))
        if not os.path.exists(os.path.join(test_dir, directory)):
            os.makedirs(os.path.join(test_dir, directory))

        subdirectory = os.path.join(in_dir, directory)

        for filename in os.listdir(subdirectory):
            file_index = int(os.path.splitext(filename)[0].split("_")[-1])

            if file_index % 20 < 10:
                # train
                shutil.copy2(
                    os.path.join(subdirectory, filename),
                    os.path.join(train_dir, directory)
                )
            else:
                shutil.copy2(
                    os.path.join(subdirectory, filename),
                    os.path.join(test_dir, directory)
                )


def csv_to_directory_structure(
        image_dir: str,
        csv_path: str,
        out_dir: str
):
    with open(csv_path, "r") as f:
        row_count = 0
        for line in f:
            image_ids = line.split(",")

            out_class_dir = os.path.join(out_dir, str(row_count))
            if not os.path.exists(out_class_dir):
                os.makedirs(out_class_dir)

            for image_id in image_ids:
                image_id = int(image_id)
                image_path = os.path.join(image_dir, f"single_organoid_{image_id}.tif")
                if os.path.isfile(image_path):
                    shutil.copy2(image_path,
                                 os.path.join(out_class_dir, f"single_organoid_{image_id}.tif"))

            row_count += 1

def copy_directory_structure():
    base_val_dir = "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/data_sets_no_garbage"
    base_csv_dir = "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/data_sets_no_garbage/clustering_results"

    val_dirs = [
        f"{base_val_dir}/C03_no_garbage",
        f"{base_val_dir}/C04_no_garbage",
        f"{base_val_dir}/C05_no_garbage",
        f"{base_val_dir}/C06_no_garbage",
        f"{base_val_dir}/C07_no_garbage",
        f"{base_val_dir}/C08_no_garbage",
    ]

    csv_paths = [
        f"{base_csv_dir}/clustering_24_10_13_17_C03_no_garbage.csv",
        f"{base_csv_dir}/clustering_24_10_13_36_C04_no_garbage.csv",
        f"{base_csv_dir}/clustering_24_10_13_55_C05_no_garbage.csv",
        f"{base_csv_dir}/clustering_24_10_14_12_C06_no_garbage.csv",
        f"{base_csv_dir}/clustering_24_10_14_19_C07_no_garbage.csv",
        f"{base_csv_dir}/clustering_24_10_14_31_C08_no_garbage.csv",
    ]

    base_out_val_dir = "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/dataset_nc"

    out_val_dirs = [
        os.path.join(base_out_val_dir, "val-C3-nc"),
        os.path.join(base_out_val_dir, "val-C4-nc"),
        os.path.join(base_out_val_dir, "val-C5-nc"),
        os.path.join(base_out_val_dir, "val-C6-nc"),
        os.path.join(base_out_val_dir, "val-C7-nc"),
        os.path.join(base_out_val_dir, "val-C8-nc"),
    ]

    for i in range(6):
        csv_to_directory_structure(
            image_dir=val_dirs[i],
            out_dir=out_val_dirs[i],
            csv_path=csv_paths[i]
        )


def create_predicted_cluster_directory(
        solution_path: str,
        data_directory: str,
        out_path: str,
        split: str = "val-C3"
):
    predicted_clusters_out_path = os.path.join(out_path, "predicted_clusters")
    truth_clusters_out_path = os.path.join(out_path, "truth_clusters")

    if not os.path.exists(predicted_clusters_out_path):
        os.makedirs(predicted_clusters_out_path)

    if not os.path.exists(truth_clusters_out_path):
        os.makedirs(truth_clusters_out_path)

    predicted_file_copy_ids = defaultdict(lambda: defaultdict(list))
    truth_file_copy_ids = defaultdict(lambda: defaultdict(list))

    # also create list of lists of predicted ids for Jannik's tool
    predicted_cluster_lists = defaultdict(list)

    with open(solution_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pred_id = row["predClusterId"]
            truth_id = row["truthClusterId"]
            truth_label = row["truthLabel"]
            image_id = row["fileId"]
            image_id = os.path.splitext(image_id)[0] + ".tif"
            image_file_path = os.path.join(data_directory, split, truth_label, image_id)

            predicted_file_copy_ids[pred_id][truth_label].append((image_file_path))
            truth_file_copy_ids[truth_label][pred_id].append((image_file_path))

            predicted_cluster_lists[pred_id].append(image_file_path.split(".")[0].split("_")[-1])

    for (pred_id, truth_to_file_ids) in predicted_file_copy_ids.items():
        num_files_in_pred = sum([len(paths) for paths in truth_to_file_ids.values()])

        directory = os.path.join(predicted_clusters_out_path, f"{pred_id} ({num_files_in_pred})")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for (truth_label, paths) in truth_to_file_ids.items():
            out_directory = os.path.join(directory, f"{truth_label} ({len(paths)}) ")
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)

            for image_path in paths:
                shutil.copy(image_path, os.path.join(out_directory, os.path.split(image_path)[-1]))

    for (truth_label, pred_to_file_ids) in truth_file_copy_ids.items():
        num_in_truth = sum([len(paths) for paths in pred_to_file_ids.values()])

        directory = os.path.join(truth_clusters_out_path, f"{truth_label} ({num_in_truth})")
        if not os.path.exists(directory):
            os.makedirs(directory)

        for (pred_id, paths) in pred_to_file_ids.items():
            out_directory = os.path.join(directory, f"{pred_id} ({len(paths)})")
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)

            for image_path in paths:
                shutil.copy(image_path, os.path.join(out_directory, os.path.split(image_path)[-1]))

    with open(os.path.join(out_path, "cluster_result"), "w") as f:
        for (pred_id, path_ids) in predicted_cluster_lists.items():
            f.write(",".join(path_ids))
            f.write("\n")


def transform_clustering_file(
        in_file: str,
        out_file: str
):
    with open(in_file, "r") as f:
        csv_reader = csv.DictReader(f)

        data = [row for row in csv_reader]

    cluster_dict = defaultdict(list)

    for row in data:
        organoid_name = row["fileId"].replace("single_organoid_", "").replace(".tif", "")

        cluster_dict[int(row['predClusterId'])].append(organoid_name)

    print(cluster_dict)

    with open(out_file, "w") as f:
        for cluster_id in range(len(cluster_dict)):
            f.write(",".join(cluster_dict[cluster_id]) + "\n")



if __name__ == "__main__":

    directories = [
        "./new-image-models/p0.0-256x256-squarePad/analysis/",
        "./new-image-models/p0.2-256x256-squarePad/analysis/",
        "./new-hist-models/model-1/analysis/"
    ]

    files = [
        "test_optimalSolution.csv",
        "test-unseen_optimalSolution.csv",
        "test-and-unseen_optimalSolution.csv"
    ]

    for directory in directories:
        for file in files:
            in_file = os.path.join(directory, file)
            out_file = os.path.join(directory, file.replace(".csv", "_formatted.csv"))

            transform_clustering_file(in_file, out_file)


    # adjust_directory_structure(
    #     in_dir="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/organoid_dataset_200",
    #     out_dir="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/split",
    #     images_per_class=20,
    # )

    # train_test_test_split(
    #     in_dir="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/cleaned",
    #     out_dir="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids New/split"
    # )

    # model_paths = [
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.0-256x256/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.0-256x256-squarePad/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.0-256x256-squarePad-res34/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.2-256x256/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.2-256x256-squarePad/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.2-256x256-res34/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.4-256x256/",
    #     # "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.4-256x256-squarePad/",
    #     "/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/models/organoids-p0.4-256x256-squarePad-res34/",
    # ]
    #
    # validation_sets = [
    #     "val-C3",
    #     "val-C4",
    #     "val-C5",
    #     "val-C6",
    #     "val-C7",
    #     "val-C8"
    # ]
    #
    # for model in model_paths:
    #     for validation_set in validation_sets:
    #         create_predicted_cluster_directory(
    #             solution_path=os.path.join(model, "analysis", f"{validation_set}_optimalSolution.csv"),
    #             split=validation_set,
    #             out_path=os.path.join(model, "analysis", f"{validation_set}_optimalSolution_clusters"),
    #             data_directory="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/dataset/"
    #         )
    #
    #         create_predicted_cluster_directory(
    #             solution_path=os.path.join(model, "analysis", f"{validation_set}_localSearchSolution.csv"),
    #             split=validation_set,
    #             out_path=os.path.join(model, "analysis", f"{validation_set}_localSearchSolution_clusters"),
    #             data_directory="/run/media/dstein/789e1bf3-b0ea-4a6a-a533-79a346a1ac3e/Organoids/dataset/"
    #         )