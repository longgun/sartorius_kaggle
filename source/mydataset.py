from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, Rotate, Compose
import numpy as np
import cv2


class Mydataset(Dataset):
    def __init__(self, df, image_path):
        self.df = df
        self.image_path = image_path
        self.images_name = self.df["id"].unique().tolist()
        self.transfrom = Compose(
            [
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Rotate(p=0.5, limit=180),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.images_name)

    def __getitem__(self, idx):
        images = self.images_name[idx]

        image_x = cv2.imread(f"{self.image_path}/{images}.png")
        mark_y = self.get_marks(self.df, images)

        augmented = self.transfrom(image=image_x, mask=mark_y)

        return augmented["image"], augmented["mask"].reshape([1, 520, 704])

    def get_marks(self, df, id):
        annotations = df["annotation"].where(df["id"] == id).dropna()
        image = np.zeros(520 * 704)
        for annotation in annotations:
            splitted = annotation.split()
            start = np.array(splitted[0::2], dtype=int)
            length = np.array(splitted[1::2], dtype=int)
            end = start + length

            for lo, hi in zip(start, end):
                image[lo - 1 : hi - 1] = 1

        return image.reshape((520, 704))
