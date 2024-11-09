import torch
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))