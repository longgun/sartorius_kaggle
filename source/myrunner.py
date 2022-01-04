import pandas as pd
from mydataset import Mydataset
import matplotlib.pyplot as plt
from mymodel import MySemiModel
import torch.nn as nn
from torch.utils.data import DataLoader, dataloader
import torch.optim as optim
import torch

import torch.nn.functional as F
import cv2
import numpy as np

from my_metrics import performance_over_thresholds


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        # self.cross = nn.BCEWithLogitsLoss(
        #    pos_weight=torch.tensor([3, 1], dtype=torch.float32)
        # )

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )
        # loss = self.cross(input, target) - torch.log(dice_loss(input, target))

        # loss = (
        #     1
        #     - (torch.log(input * torch.log(target)) / 2)
        #     - torch.log(dice_loss(input, target))
        # )

        # loss = self.alpha * self.focal(input, target)

        return loss.mean()


class MyFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        loss = 0

        return loss


class MyRunner:
    def __init__(self, train_csv):
        self.raw_path_train = train_csv
        self.image_path_train = "../data/train"

    def main(self):
        df_train = pd.read_csv(self.raw_path_train)
        ds_train = Mydataset(df_train, self.image_path_train)
        dl_train = DataLoader(ds_train, batch_size=12, pin_memory=True, shuffle=False)

        model = MySemiModel()
        print(model)
        model.cuda()
        model.train()
        criterion = MixedLoss(10, 2)
        # criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        optimizer.zero_grad()

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        for epoch in range(30):
            print(epoch)
            total_loss = 0.0
            for i, batch in enumerate(dl_train):
                image, mark = batch
                image, mark = image.cuda(), mark.cuda()
                image = image.float()
                mark = mark.float()

                predicted = model(image)
                loss = criterion(predicted, mark)

                # print(performance_over_thresholds(mark, predicted))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(total_loss / i)

        torch.save(model.state_dict(), "./model.pth")

        model.eval()

        image, mark = next(iter(ds_train))
        plt.imshow(mark[0])
        plt.savefig("test_origin.png")

        predicted = model(image.unsqueeze(0).cuda().float())
        predicted = predicted[0, 0].cpu().detach().numpy()
        predicted = cv2.threshold(predicted, 0.5, 1, cv2.THRESH_BINARY)[1]
        plt.imshow(predicted)
        plt.savefig("test.png")


def post_process(probability, threshold=0.5, min_size=300):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = []
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            a_prediction = np.zeros((520, 704), np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions


if __name__ == "__main__":
    OBJ = MyRunner("../data/train.csv")
    OBJ.main()
