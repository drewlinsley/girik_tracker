import torch.nn as nn
import torch.nn.functional as F
import torch
try:
    from performer_pytorch import Performer
except:
    print("Failed to import Performer.")


class PerformerModel(nn.Module):

    def __init__(self, dimensions, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt'):
        '''
        '''
        super(PerformerModel, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.preproc = nn.Conv3d(3, dimensions, kernel_size=1, padding=1 // 2)
        self.Performer = Performer(
            dim = dimensions,
            depth = 1,
            heads = 4,
            causal = True
        )

        self.readout_conv = nn.Conv2d(dimensions, 1, 1)
        self.target_conv = nn.Conv2d(dimensions + 1, 1, 5, padding=5 // 2)
        torch.nn.init.zeros_(self.target_conv.bias)
        self.readout_dense = nn.Linear(1, 1)
        self.nl = F.softplus

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run Performer
        # excitation = self.Performer(xbn.permute(0, 2, 3, 4, 1).reshape(xbn.shape[0], -1, xbn.shape[1]))
        excitation = self.Performer(xbn.reshape(xbn.shape[0], -1, xbn.shape[1]))
        output = excitation.reshape(xbn.shape).mean(2)
        output = torch.cat([output, x[:, 2, 0][:, None]], 1)

        # Potentially combine target_conv + readout_bn into 1
        output = self.target_conv(output)  # output.sum(1, keepdim=True))  # 2 channels -> 1. Is the dot in the target?
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(xbn.shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)
        return output, jv_penalty

