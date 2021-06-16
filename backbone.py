import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, config, **kwargs):
        super(LSTM, self).__init__()
        input_size = config['model']['LSTM']['input_size']
        hidden_size = config['model']['LSTM']['hidden_size']
        output_size = config['model']['LSTM']['output_size']
        bidirectional = config['model']['LSTM']['bidirectional']
        act = config['model']['LSTM']['activation']

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=config['model']['LSTM']['num_layers'],
                            batch_first=True, bidirectional=bidirectional)
        self.scaling_layer = nn.Sequential(
            nn.Linear(max(1, int(bidirectional) * 2)
                      * hidden_size, output_size),
            eval(f'nn.{act}()'))
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'scaling_layer.0.weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

    def forward(self, features, **kwargs):
        predicted, _ = self.lstm(features)
        predicted = self.scaling_layer(predicted)
        return predicted