from random import shuffle
from pathlib import Path
from shutil import copy2

import pandas as pd


def pub_folder_to_csv(folder_path):
    """Take the excerpt from the original train.csv that corresponds to publications located in folder_path
    args:
        folder_path: path to the folder that contains the publications subset.
    """
    files = [f.stem for f in folder_path.iterdir()]
    files_df = pd.DataFrame(files, columns=["Id"])
    df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
    df = df.merge(files_df)
    df.to_csv(folder_path.parent / (folder_path.name + ".csv"), index=False)


def downsample(file_path):
    """Reduce the number of sentences without named entities.
    args:
        file_path: path to the json file with all sentences from all publications
    """
    df = pd.read_json(file_path, orient="records", lines=True)
    entities = df[df.ent_count > 0]
    n_sample = 990  # entities.ent_count.sum()
    no_entities_subset = df[df.ent_count == 0].sample(n_sample)
    entities.append(no_ent_subset).sort_index().set_index("Id").to_json(
        file_path.parent / (file_path.stem + "_downsampled.json"),
        orient="records",
        lines=True,
    )


# fonction pour diviser le subset en un ensemble d'apprentissage et un ensemble de validation de sorte à avoir
# /!\ on split selon la colonne dataset_label et pas dataset_title ce qui veut dire que des dataset_titles vont se retrouver à la fois en train et en valid (avec des labels différents, ce qui peut être intéressant)
# on split au niveau des publications pas des phrases, certaines publications ont à la fois des datasets de train et de val, on les met dans val.
# TODO splitter au niveau des phrases.
def dataset_split(file_path, publications_path, val_size=0.1):
    df = pd.read_csv(file_path)
    labels = df.dataset_label.unique()
    shuffle(labels)
    split = int(len(labels) * val_size)
    train_labels = labels[split:]
    val_labels = labels[:split]
    train_df = df[df.dataset_label.isin(train_labels)]
    val_df = df[df.dataset_label.isin(val_labels)]

    train_ids = set(train_df.Id).difference(set(val_df.Id))
    val_ids = set(val_df.Id)
    intersection_ids = set(train_df.Id).intersection(val_ids)
    intersection_df = df[df.Id.isin(intersection_ids)]
    (publications_path.parent / "train").mkdir(exist_ok=True)
    (publications_path.parent / "val").mkdir(exist_ok=True)

    for p in train_ids:
        copy2(
            publications_path / (p + ".json"),
            publications_path.parent / "train" / (p + ".json"),
        )
    for p in val_ids:
        copy2(
            publications_path / (p + ".json"),
            publications_path.parent / "val" / (p + ".json"),
        )
    with open(publications_path.parent / "split.txt", "w") as f:
        f.write("Train labels: " + str(len(train_labels)))
        f.write(
            "\nTrain labels that appear in val: "
            + str(
                len(set(train_labels).intersection(set(intersection_df.dataset_label)))
            )
        )
        f.write("\nTrain pubs: " + str(len(train_ids)))
        f.write("\nTrain dataset mentions: " + str(len(df[df.Id.isin(train_ids)])))
        f.write("\nVal only labels: " + str(len(val_labels)))
        f.write("\nVal pubs: " + str(len(val_ids)))
        f.write("\nVal dataset mentions: " + str(len(df[df.Id.isin(val_ids)])))
        f.write(
            "\nVal pubs with datasets that appear in train: "
            + str(len(intersection_ids))
        )
        f.write(
            "\nVal mentions of dataset that appear in train: "
            + str(len(intersection_df))
        )
        f.write(
            "\n/!\ Mentions reported in the train.csv, only 40% of the actual number of mentions of identified datasets"
        )


if __name__ == "__main__":
    # pub_folder_to_csv(Path("../input/subset/data/train"))
    # pub_folder_to_csv(Path("../input/subset/data/val"))
    pub_folder_to_csv(Path("../input/subset_dataset-split/publications"))
    dataset_split(
        Path("../input/subset_dataset-split/publications.csv"),
        Path("../input/subset_dataset-split/publications"),
    )
