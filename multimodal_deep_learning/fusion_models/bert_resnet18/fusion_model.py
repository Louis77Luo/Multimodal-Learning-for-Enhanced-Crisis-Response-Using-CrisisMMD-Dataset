from utils import device
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()

        self.fc1_1 = nn.Linear(768, 1024)
        self.fc1_2 = nn.Linear(1024, 512)
        self.fc1_3 = nn.Linear(512, 128)

        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv2_5 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.conv2_6 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.fc2_1 = nn.Linear(256 * 8, 256)
        self.fc2_2 = nn.Linear(256, 127)

        self.fc_final1 = nn.Linear(256, 256)
        self.fc_final2 = nn.Linear(256, 128)
        self.fc_final3 = nn.Linear(128, 1)

    def forward(self, input1, input2, input3):
        x1 = F.relu(self.fc1_1(input1))
        x1 = F.relu(self.fc1_2(x1))
        x1 = F.relu(self.fc1_3(x1))

        x2 = input2.unsqueeze(1)
        x2 = F.relu(self.conv2_1(x2))
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.relu(self.conv2_3(x2))
        x2 = F.relu(self.conv2_4(x2))
        x2 = F.relu(self.conv2_5(x2))
        x2 = F.relu(self.conv2_6(x2))
        x2 = x2.view(x2.size(0), -1)
        x2 = F.relu(self.fc2_1(x2))
        x2 = F.relu(self.fc2_2(x2))

        input3 = input3.view(-1, 1)
        x = torch.cat((x1, x2, input3), dim=1)

        x = F.relu(self.fc_final1(x))
        x = F.relu(self.fc_final2(x))
        x = torch.sigmoid(self.fc_final3(x))
        return x


if __name__ == '__main__':
    model = FusionModel()
    model.to(device)
    summary(model, [(768,), (512,), (1,)])
