import numpy as np
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F

# 繰り返す回数
epochSize = 500

batchSize = 5

class SnowDepthPredictor(nn.Module):
    def __init__(self, input_size=13, output_size=1):
        super(SnowDepthPredictor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)

neuralnetwork = SnowDepthPredictor
