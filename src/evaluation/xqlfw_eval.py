
import sys
sys.path.append("./..")

import os
import pickle

from src.calculate_qs import __calculate_score_batch__ as FaceQAN


class XQLFWImageDataset():

    def __init__(self) -> None:
        
        self.loc = os.path.join(os.path.dirname(__file__), "..", "datasets", "lfw_original_imgs_min_qual0.85variant11")

        self.items = []
        for (dir, subdirs, files) in os.walk(self.loc):
            self.items.extend(list(filter(lambda loc: loc.endswith(".jpg"), map(lambda file: os.path.join(dir, file), files))))

    def __getitem__(self, x) -> str:
        return self.items[x]

    def __len__(self) -> int:
        return len(self.items)


if __name__ == "__main__":

    assert os.path.exists(os.path.join(os.path.dirname(__file__), "..", "datasets", "xqlfw_aligned_112")), f" Place the extracted XQLFW aligned data into ./datasets"

    print(f" => Running FaceQAN over XQLFW dataset, with standard values of hyperparameters.")

    xqlfw_images = XQLFWImageDataset()

    xqlfw_scores = FaceQAN(list(xqlfw_images), eps=0.001, l=5, k=10, p=5)

    print(f" => Saving quality scores to {os.path.join(os.path.dirname(__file__), 'xqlfw_results.pkl')}")

    with open(os.path.join(os.path.dirname(__file__), "xqlfw_results.pkl"), "wb") as pkl_out:
        pickle.dump(xqlfw_scores, pkl_out)
    