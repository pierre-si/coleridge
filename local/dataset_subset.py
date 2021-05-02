from pathlib import Path
import pandas as pd


def pub_folder_to_csv(folder_path):
    files = [f.stem for f in folder_path.iterdir()]
    files_df = pd.DataFrame(files, columns=["Id"])
    df = pd.read_csv("../input/coleridgeinitiative-show-us-the-data/train.csv")
    df = df.merge(files_df)
    df.to_csv(folder_path.parent / (folder_path.name + ".csv"), index=False)


# script utilisé pour créer le train.csv de input/subset (sous-ensemble de publications)
if __name__ == "__main__":
    pub_folder_to_csv(Path("../input/subset/data/train"))
    pub_folder_to_csv(Path("../input/subset/data/val"))
