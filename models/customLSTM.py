import torch
import numpy as np
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, in_channels):
        super(LSTM, self).__init__()

        def lstm_block(input_size):
            in_linear1_size = input_size
            in_linear2_size = input_size // 2
            in_lstm_size = np.sqrt(input_size)
            return nn.Sequential(
                nn.Linear(in_linear1_size,in_linear2_size),
                nn.Linear(in_linear2_size, in_lstm_size),
                nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=1)
            )

        self.lstm = lstm_block(input_size=input_size)


    def forward(self, x):
        dec = self.lstm(x)
        
        # Add padding if necessary
        target_h, target_w = x.size(2), x.size(3)
        pad_h = target_h - dec.size(2)
        pad_w = target_w - dec.size(3)

        if pad_h > 0 or pad_w > 0:
            dec = nn.functional.pad(dec, (0, pad_w, 0, pad_h))

        return dec