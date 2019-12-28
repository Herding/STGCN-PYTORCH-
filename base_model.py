from layers import *
from os.path import join as pjoin
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_his, n_route, Ks, Kt, blocks, g_kernel, drop_prob):
        super(Model, self).__init__()

        self.Ko = n_his
        self.train_loss = nn.MSELoss()

        self.stconv1 = STGCNBlock(Ks, Kt, blocks[0], [n_route, blocks[0][-1]], g_kernel, drop_prob)
        self.Ko -= 2 * (Ks - 1)
        self.stconv2 = STGCNBlock(Ks, Kt, blocks[1], [n_route, blocks[1][-1]], g_kernel, drop_prob)
        self.Ko -= 2 * (Ks - 1)

        if self.Ko > 1:
            self.output = Output(self.Ko, blocks[1][-1], [n_route, blocks[1][-1]])
        else:
            raise ValueError(f'ERROR: kernel size Ko must be greater than 1, but received "{self.Ko}".')

    def forward(self, x):
        x = self.stconv1(x)
        x = self.stconv2(x)
        return self.output(x)

    def save(self, model_name, save_path='./output/models/'):
        path = pjoin(save_path, model_name)
        torch.save(self.state_dict(), path)
        print(f'<< Saving model to {path} ...')
