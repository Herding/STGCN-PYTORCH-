import torch
import torch.nn as nn


class TCL(nn.Module):
    def __init__(self, Kt, c_in, c_out, act_func='relu'):
        super(TCL, self).__init__()

        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.act_func = act_func

        self.conv = nn.Conv2d(c_in, c_out, (1, 1), stride=1)

        if act_func == 'GLU':
            self.tconv = nn.Conv2d(c_in, 2 * c_out, (Kt, 1), stride=1)
            self.sigmoid = nn.Sigmoid()
        else:
            self.tconv = nn.Conv2d(c_in, c_out, (Kt, 1), stride=1)
            if act_func == 'sigmoid':
                self.sigmoid = nn.Sigmoid()
            elif act_func == 'relu':
                self.relu = nn.ReLU()

    def forward(self, x):
        _, _, T, n = x.shape

        if self.c_in > self.c_out:
            x_input = self.conv(x)
        elif self.c_in < self.c_out:
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.c_out - self.c_in, T, n])], dim=1)
        else:
            x_input = x

        x_input = x_input[:, :, self.Kt - 1: T, :]

        if self.act_func == 'GLU':
            x_tconv = self.tconv(x)
            return (x_tconv[:, 0: self.c_out, :, :] + x_input) * self.sigmoid(x_tconv[:, -self.c_out:, :, :])
        else:
            x_tconv = self.tconv(x)
            if self.act_func == 'linear':
                return x_tconv
            elif self.act_func == 'sigmoid':
                return self.sigmoid(x_tconv)
            elif self.act_func == 'relu':
                return self.relu(x_tconv + x_input)
            else:
                raise ValueError(f'ERROR: activation function "{self.act_func}" is not defined.')


class SCL(nn.Module):
    def __init__(self, ks, c_in, c_out, kernel):
        super(SCL, self).__init__()

        self.Ks = ks
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.ws = nn.Parameter(torch.randn(ks * c_in, c_out), requires_grad=True)
        self.bs = nn.Parameter(torch.zeros(c_out), requires_grad=True)

        self.conv = nn.Conv2d(c_in, c_out, (1, 1), stride=1)
        self.relu = nn.ReLU()

    def gconv(self, x, T, n):
        x = torch.reshape(x.permute(0, 2, 3, 1), [-1, n, self.c_in])
        n = self.kernel.shape[0]
        x_tmp = torch.reshape(x.permute(0, 2, 1), [-1, n])
        x_mul = torch.reshape(torch.mm(x_tmp, self.kernel), [-1, self.c_in, self.Ks, n])
        x_ker = torch.reshape(x_mul.permute(0, 3, 1, 2), [-1, self.c_in * self.Ks])
        x = torch.reshape(torch.mm(x_ker, self.ws), [-1, n, self.c_out]) + self.bs
        x = torch.reshape(x, [-1, T, n, self.c_out])
        return x.permute(0, 3, 1, 2)

    def forward(self, x):
        _, _, T, n = x.shape

        if self.c_in > self.c_out:
            x_input = self.conv(x)
        elif self.c_in < self.c_out:
            x_input = torch.cat([x, torch.zeros([x.shape[0], self.c_out - self.c_in, T, n])], dim=1)
        else:
            x_input = x

        x_gconv = self.gconv(x, T, n)
        return self.relu(x_gconv[:, 0: self.c_out, :, :] + x_input)


class Output(nn.Module):
    def __init__(self, T, channel, norm_dims, act_func='GLU'):
        super(Output, self).__init__()
        self.T = T
        self.act_func = act_func

        self.tcl1 = TCL(self.T, channel, channel, self.act_func)
        self.norm = nn.LayerNorm(norm_dims)
        self.tcl2 = TCL(1, channel, channel, 'sigmoid')
        self.fcon = nn.Conv2d(channel, 1, (1, 1), stride=1)

    def forward(self, x):
        x = self.tcl1(x)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = self.tcl2(x.permute(0, 3, 1, 2))
        return self.fcon(x)


class STGCNBlock(nn.Module):
    def __init__(self, Ks, Kt, channels, norm_dims, g_kernel, drop_prob, act_func="GLU"):
        super(STGCNBlock, self).__init__()

        self.Ks = Ks
        self.Kt = Kt
        c_si, c_t, c_oo = channels

        self.tcl1 = TCL(Kt, c_si, c_t, act_func=act_func)
        self.scl = SCL(Ks, c_t, c_t, g_kernel)
        self.tcl2 = TCL(Kt, c_t, c_oo)
        self.norm = nn.LayerNorm(norm_dims)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.tcl1(x)
        x = self.scl(x)
        x = self.tcl2(x)
        x = self.norm(x.permute(0, 2, 3, 1))
        return self.dropout(x.permute(0, 3, 1, 2))
