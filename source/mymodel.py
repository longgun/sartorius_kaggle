import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch


class UpModule(nn.Module):
    def __init__(self, start, end) -> None:
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(start, end, kernel_size=(2, 2), stride=2)
        self.extract1 = nn.Conv2d(start, end, kernel_size=(3, 3), padding="same")
        self.extract2 = nn.Conv2d(end, end, kernel_size=(3, 3), padding="same")

    def forward(self, input1, input2):
        out1 = F.relu(self.convTrans(input1))
        out2 = F.relu(self.extract1(torch.concat([input2, out1], dim=1)))

        return F.relu(self.extract2(out2))


class DownModule(nn.Module):
    def __init__(self, start, end, flag=True) -> None:
        super().__init__()
        self.flag = flag
        self.contract1 = nn.Conv2d(start, end, kernel_size=(3, 3), padding="same")
        self.contract2 = nn.Conv2d(end, end, kernel_size=(3, 3), padding="same")
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, input):
        out1 = F.relu(self.contract1(input))
        out2 = F.relu(self.contract2(out1))
        out3 = F.relu(self.maxpool1(out2))

        if self.flag:
            pool = out2
            out = out3
        else:
            pool = out2
            out = out2

        return pool, out


class MyDown(nn.Module):
    def __init__(self, start) -> None:
        super().__init__()

        self.down1 = DownModule(start, 16)
        self.down2 = DownModule(16, 32)
        self.down3 = DownModule(32, 64)

    def forward(self, input):
        pool1, down1 = self.down1(input)
        pool2, down2 = self.down2(down1)
        pool3, down3 = self.down3(down2)

        return pool1, pool2, pool3, down3


class MyUp(nn.Module):
    def __init__(self, end) -> None:
        super().__init__()

        self.upt1 = UpModule(128, 64)
        self.upt2 = UpModule(64, 32)
        self.upt3 = UpModule(32, end)

    def forward(self, input, down1, down2, down3):
        upt1 = self.upt1(input, down3)
        upt2 = self.upt2(upt1, down2)
        upt3 = self.upt3(upt2, down1)

        return upt3


class MySemiModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.down = MyDown(3)
        self.flat = DownModule(64, 128, False)
        self.up = MyUp(16)
        self.out = nn.ConvTranspose2d(16, 1, kernel_size=(1, 1))

    def forward(self, input):
        pool1, pool2, pool3, down = self.down(input)
        _, flat = self.flat(down)
        up = self.up(flat, pool1, pool2, pool3)

        out = self.out(up)

        return out


# class MySemiModel(nn.Module):
#     def __init__(self):
#         super(MySemiModel, self).__init__()

#         self.contracting1_1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding="same")
#         self.contracting1_2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same")
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

#         self.contracting2_1 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding="same")
#         self.contracting2_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same")
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

#         self.contracting3_1 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding="same")
#         self.contracting3_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding="same")

#         self.upconv0 = nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=2)

#         self.expanding1_1 = nn.Conv2d(64, 32, kernel_size=(3, 3), padding="same")
#         self.expanding1_2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding="same")

#         self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=(2, 2), stride=2)

#         self.expanding2_1 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding="same")
#         self.expanding2_2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same")
#         self.expanding2_3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding="same")

#         self.out = nn.ConvTranspose2d(16, 1, kernel_size=(1, 1))

#     def forward(self, image, flg="labeled"):
#         down1 = F.relu(self.contracting1_1(image))
#         down2 = F.relu(self.contracting1_2(down1))
#         pool1 = F.relu(self.maxpool1(down2))

#         down3 = F.relu(self.contracting2_1(pool1))
#         down4 = F.relu(self.contracting2_2(down3))
#         pool2 = F.relu(self.maxpool2(down4))

#         down5 = F.relu(self.contracting3_1(pool2))
#         down6 = F.relu(self.contracting3_2(down5))

#         upconv0 = F.relu(self.upconv0(down6))
#         up1 = F.relu(self.expanding1_1(torch.concat([upconv0, down4], dim=1)))
#         # up1 = F.relu(self.expanding1_1(upconv0))
#         up2 = F.relu(self.expanding1_2(up1))

#         upconv1 = F.relu(self.upconv1(up2))
#         up3 = F.relu(self.expanding2_1(torch.concat([upconv1, down2], dim=1)))
#         # up3 = F.relu(self.expanding2_1(upconv1))
#         up4 = F.relu(self.expanding2_2(up3))

#         # up4 = F.pad(up4, (12, 16, 12, 16))
#         out = self.out(up4)

#         return out
