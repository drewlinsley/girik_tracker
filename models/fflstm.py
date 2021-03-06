import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)


class FFLSTM(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        '''
        Feedforward hGRU with input layer initialised with gaussian weights 
        (not learnable - no grad), then upsampled to 8 feature maps, and 
        fed to hGRU cell
        '''
        super(FFLSTM, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.batch_size=2
        self.hgru_size=4
        self.embedding_dim=3
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv00 = nn.Conv3d(3, self.embedding_dim, kernel_size=7, bias=False, padding=3)
        nn.init.normal_(self.conv00.weight, mean=0.0, std=1.0)
        self.conv0 = nn.Conv3d(1, 8, kernel_size=7, bias=False, padding=3)
        # part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        # part1 = np.expand_dims(part1, axis=0)
        # print(part1.shape)
        # import pdb; pdb.set_trace()
        # self.conv00.weight.data = torch.FloatTensor(part1)

        # self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        # self.unit1 = hConvGRUCell(self.hgru_size, self.hgru_size, filt_size)
        self.unit1 = nn.LSTM(3, self.hgru_size, num_layers=2, bidirectional=True)#, batch_first=True)
        print("Training with filter size:", filt_size, "x", filt_size)
        # self.bn = nn.BatchNorm3d(self.hgru_size, eps=1e-03, track_running_stats=False)
        self.bn = nn.InstanceNorm3d(self.hgru_size*2, eps=1e-03, track_running_stats=False)
        # self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        # init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        # self.fc4 = nn.Linear(1*self.hgru_size*65*65, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.fc4 = nn.Linear(1*self.hgru_size*2*32*64*64, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.avgpool = nn.AvgPool3d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        # z-score normalize input video to the mean and std of the gaussian weight inits
        # x = (x - torch.mean(x, axis=[1, 3, 4], keepdims=True)) / torch.std(x, axis=[1, 3, 4], keepdims=True)
        # average across the RGB channel dimension
        # x=torch.mean(x,axis=1, keepdims=True)

        with torch.no_grad():   # stopping the first gausian init input layer from learning 
            out = self.conv00(x)
            # out = self.conv0(out) # 1x1 conv to inflate feature maps to 8 dims
            # out=out.repeat(1,self.hgru_size,1,1,1)
        # import pdb; pdb.set_trace()
        # out.requires_grad=True
        out = torch.pow(out, 2)
        # out = out.permute(2,0,1,3,4)
        # internal_state = torch.zeros_like(out, requires_grad=False)
        # internal_state = torch.zeros_like(torch.empty(4,4,8))
        # internal_state = torch.zeros_like(torch.empty(4,4,64,64,64).cuda(), requires_grad=False)

        # for t in range(0,out.shape[0]):
        for t in range(0,self.timesteps):
            if t==0:
                output, (h_n, c_n) = self.unit1(out.view(-1, self.batch_size,self.embedding_dim))
            else:
                output, (h_n, c_n) = self.unit1(out.view(-1, self.batch_size,self.embedding_dim), (h_n, c_n))
            # internal_state, g2t = self.unit1(out, internal_state, timestep=t)
            # internal_state, g2t = self.unit1(out.view(-1, self.batch_size,self.embedding_dim), internal_state)
            # if t == self.timesteps - 2:
            #     state_2nd_last = internal_state
            # elif t == self.timesteps - 1:
            #     last_state = internal_state        

        # import pdb; pdb.set_trace()
        # output = self.bn(internal_state)
        output = self.bn(output.view(self.batch_size, self.hgru_size*2, 64, 128, 128))
        # output = torch.mean(output,1)
        output = self.avgpool(output)
        output=output.view(self.batch_size,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        # pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        # mu = 0.9
        # double_neg = False
        # if self.training and self.jacobian_penalty:
        #     if pen_type == 'l1':
        #         norm_1_vect = torch.ones_like(last_state)
        #         norm_1_vect.requires_grad = False
        #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        #                                       create_graph=self.jacobian_penalty, allow_unused=True)[0]
        #         jv_penalty = (jv_prod - mu).clamp(0) ** 2
        #         if double_neg is True:
        #             neg_norm_1_vect = -1 * norm_1_vect.clone()
        #             jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
        #                                           create_graph=True, allow_unused=True)[0]
        #             jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
        #             jv_penalty = jv_penalty + jv_penalty2
        #     elif pen_type == 'idloss':
        #         norm_1_vect = torch.rand_like(last_state).requires_grad_()
        #         jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
        #                                       create_graph=True, allow_unused=True)[0]
        #         jv_penalty = (jv_prod - norm_1_vect) ** 2
        #         jv_penalty = jv_penalty.mean()
        #         if torch.isnan(jv_penalty).sum() > 0:
        #             raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, loss

