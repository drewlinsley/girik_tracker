import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        # import pdb; pdb.set_trace()

        state_2nd_last, last_state = ctx.saved_tensors
        
        args = ctx.args
        truncate_iter = args[-1]
        exp_name = args[-2]
        i = args[-3]
        epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        normsg.append(normg.data.item())
        normsv.append(normg.data.item())
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            
            if normg > 1 or normv > normsv[-1] or normv < 1e-9:
                normsg.append(normg.data.item())
                normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            
            normsv.append(normv.data.item())
            normsg.append(normg.data.item())
        
        return (None, neumann_g, None, None, None, None)


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, timesteps=8, grad_method='bptt'):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, kernel_size, kernel_size))
        
        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        # self.bn = nn.ModuleList([nn.BatchNorm2d(25, eps=1e-03, affine=True, track_running_stats=False) for i in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(4)])

        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
        
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
        
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data = -self.u1_gate.bias.data
        #self.outscale = nn.Parameter(torch.tensor([8.0]))
        #self.outintercpt = nn.Parameter(torch.tensor([-4.0]))
        self.softpl = nn.Softplus()
        self.softpl.register_backward_hook(lambda module, grad_i, grad_o: print(len(grad_i)))

    def forward(self, input_, prev_state2, timestep=0):
        activ = F.softplus
        #activ = torch.sigmoid
        #activ = torch.tanh
        g1_t = torch.sigmoid((self.u1_gate(prev_state2)))
        c1_t = self.bn[1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
        
        # import pdb; pdb.set_trace()
        
        next_state1 = activ(input_ - activ(c1_t * (self.alpha * prev_state2 + self.mu)))
        
        g2_t = torch.sigmoid((self.u2_gate(next_state1)))
        c2_t = self.bn[3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
        
        h2_t = activ(self.kappa * next_state1 + self.gamma * c2_t + self.w * next_state1 * c2_t)
        prev_state2 = (1 - g2_t) * prev_state2 + g2_t * h2_t

        prev_state2 = F.softplus(prev_state2)

        return prev_state2, g2_t


class FFhGRU1(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):

        super().__init__()
        filter_size = filt_size
        padding_size = filter_size // 2
        
        self.conv0 = nn.Conv3d(3, 25, kernel_size=7, bias=False, padding=3)
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        # part1 = np.load('utils/gabor_serre.npy') # problems with dimensions (25,1,7,7). Confirm with Drew which dimensions to be inflated for videos.
        # self.conv0.weight.data = torch.FloatTensor(part1)


        self.bn0 = nn.BatchNorm3d(25)
        # kernel = np.load("gabor_serre.npy")
        # self.conv0.weight.data = torch.FloatTensor(kernel)
        
        self.conv1 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn1 = nn.BatchNorm3d(25)
        
        self.conv2 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn2 = nn.BatchNorm3d(25)
        
        self.conv3 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn3 = nn.BatchNorm3d(25)
        
        self.conv4 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn4 = nn.BatchNorm3d(25)
        
        self.conv5 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn5 = nn.BatchNorm3d(25)

        self.conv6 = nn.Conv3d(25, 25, kernel_size=filter_size, padding=padding_size)
        self.bn6 = nn.BatchNorm3d(25)

        self.conv7 = nn.Conv3d(25, 2, kernel_size=1)
        
        self.bn7 = nn.BatchNorm3d(25)
        # self.fc = nn.Linear(2, 2)
        # self.fc1 = nn.Linear(65536, 128)
        
        self.fc1 = nn.Linear(128, 2*2*64*128)
        self.fc2 = nn.Linear(2*2*64*128, 128)
        self.fc3 = nn.Linear(128, 2)

        self.fc4 = nn.Linear(2*64*128*128, 2) # the first 2 is for batch size
        self.fc5 = nn.Linear(2,1)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out = self.conv0(x)
        out = self.bn0(out)
        #out = torch.pow(out, 2)
        #print(out.shape)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        #print(out.shape)
        out = self.conv5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = self.conv6(out)
        out = self.bn6(out)
        out = F.relu(out)
        # output = self.conv7(out)
        out = self.conv7(out)
        print(out.shape)

        # out = self.fc1(out)
        # out = self.fc2(out)
        # output = self.fc3(out)
        out=out.view(2,-1)
        output=self.fc4(out)
        # output=self.fc5(out)

        loss = criterion(output, target)
        jv_penalty = torch.tensor([1]).float().cuda()
        
        return output, jv_penalty, loss

class FFhGRU(nn.Module):

    def __init__(self, batch_size, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        '''
        Feedforward hGRU with input layer initialised with gaussian weights 
        (not learnable - no grad), then upsampled to 8 feature maps, and 
        fed to hGRU cell
        ADDED DROPOUT
        '''
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.batch_size=int(batch_size/torch.cuda.device_count())
        self.hgru_size=4
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv00 = nn.Conv3d(3, 1, kernel_size=5, bias=False, padding=2)
        nn.init.normal_(self.conv00.weight, mean=5.0, std=2.0)
        # self.conv0 = nn.Conv3d(1, 8, kernel_size=7, bias=False, padding=3)
        # part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        # part1 = np.expand_dims(part1, axis=0)
        # print(part1.shape)
        # import pdb; pdb.set_trace()
        # self.conv00.weight.data = torch.FloatTensor(part1)

        # self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        self.unit1 = hConvGRUCell(self.hgru_size, self.hgru_size, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        # init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        # self.fc4 = nn.Linear(1*self.hgru_size*64*64, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.fc4 = nn.Linear(1024, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        # z-score normalize input video to the mean and std of the gaussian weight inits
        # x = (x - torch.mean(x, axis=[1, 3, 4], keepdims=True)) / torch.std(x, axis=[1, 3, 4], keepdims=True)
        # average across the RGB channel dimension
        # x=torch.mean(x,axis=1, keepdims=True)
        # import pdb; pdb.set_trace()
        # x=x.permute(0,2,1,3,4)
        # x = (x - torch.mean(x, axis=[2, 3, 4])) / torch.std(x, axis=[2, 3, 4])

        # for i in range(x.shape[0]):
        #     import pdb; pdb.set_trace()
        #     # mu=torch.mean(x[i])
        #     # sigma=torch.std(x[i])
        #     # x[i]=(x[i]-mu)/sigma
        #     x[i] = (x[i] - torch.mean(x[i], axis=[0, 2, 3])) / torch.std(x[i], axis=[0, 2, 3])

        # out=out.permute(0,2,3,4,1)
        # npnp=out.cpu().detach().numpy()
        # np.save("gaussian_responses.npy", npnp)

        with torch.no_grad():   # stopping the first gausian init input layer from learning 
            out = self.conv00(x)
            # import pdb; pdb.set_trace()
            # out = self.conv0(out) # 1x1 conv to inflate feature maps to 8 dims
            out=out.repeat(1,self.hgru_size,1,1,1)
        # out.requires_grad=True
        # out = F.dropout(out, p=0.5, training=self.training)
        out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        internal_state = torch.zeros_like(out[0], requires_grad=False)
        # import pdb;pdb.set_trace()

        for t in range(0,out.shape[0]):
            y=out[t]
            internal_state, g2t = self.unit1(y, internal_state, timestep=t)
            if t == self.timesteps - 2:
                state_2nd_last = internal_state
            elif t == self.timesteps - 1:
                last_state = internal_state        

        # output = F.dropout(internal_state, p=0.5, training=self.training)
        output = self.bn(internal_state)
        # output = torch.mean(output,1)
        output = self.avgpool(output)
        # import pdb; pdb.set_trace()
        output=output.view(self.batch_size,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
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

class FFhGRUdown(nn.Module):

    def __init__(self, batch_size, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        '''
        Feedforward hGRU with input layer initialised with gaussian weights 
        (not learnable - no grad), then upsampled to 8 feature maps, and 
        fed to hGRU cell
        Executed this one on x7 as not parallel program
        '''
        super(FFhGRUdown, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.batch_size=batch_size
        self.hgru_size=4
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv00 = nn.Conv3d(3, 1, kernel_size=5, bias=False, padding=2)
        nn.init.normal_(self.conv00.weight, mean=0.0, std=2.0)
        # self.conv0 = nn.Conv3d(1, 8, kernel_size=7, bias=False, padding=3)
        # part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        # part1 = np.expand_dims(part1, axis=0)
        # print(part1.shape)
        # import pdb; pdb.set_trace()
        # self.conv00.weight.data = torch.FloatTensor(part1)

        # self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        self.unit1 = hConvGRUCell(self.hgru_size, self.hgru_size, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        # init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        self.fc4 = nn.Linear(1*self.hgru_size*64*64, 1) # the first 2 is for batch size, the second digit is for the dimension
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        # z-score normalize input video to the mean and std of the gaussian weight inits
        # x = (x - torch.mean(x, axis=[1, 3, 4], keepdims=True)) / torch.std(x, axis=[1, 3, 4], keepdims=True)
        # average across the RGB channel dimension
        # x=torch.mean(x,axis=1, keepdims=True)
        # import pdb; pdb.set_trace()
        # x=x.permute(0,2,1,3,4)
        # x = (x - torch.mean(x, axis=[2, 3, 4])) / torch.std(x, axis=[2, 3, 4])

        # for i in range(x.shape[0]):
        #     import pdb; pdb.set_trace()
        #     # mu=torch.mean(x[i])
        #     # sigma=torch.std(x[i])
        #     # x[i]=(x[i]-mu)/sigma
        #     x[i] = (x[i] - torch.mean(x[i], axis=[0, 2, 3])) / torch.std(x[i], axis=[0, 2, 3])

        # out=out.permute(0,2,3,4,1)
        # npnp=out.cpu().detach().numpy()
        # np.save("gaussian_responses.npy", npnp)

        with torch.no_grad():   # stopping the first gausian init input layer from learning 
            out = self.conv00(x)
            # import pdb; pdb.set_trace()
            # out = self.conv0(out) # 1x1 conv to inflate feature maps to 8 dims
            out=out.repeat(1,self.hgru_size,1,1,1)
        # out.requires_grad=True
        out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        internal_state = torch.zeros_like(out[0], requires_grad=False)
        # import pdb;pdb.set_trace()

        for t in range(0,out.shape[0]):
            y=out[t]
            internal_state, g2t = self.unit1(y, internal_state, timestep=t)
            if t == self.timesteps - 2:
                state_2nd_last = internal_state
            elif t == self.timesteps - 1:
                last_state = internal_state        

        output = self.bn(internal_state)
        # output = torch.mean(output,1)
        output = self.avgpool(output)
        # import pdb; pdb.set_trace()
        output=output.view(self.batch_size,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
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



class FFhGRUwithoutGaussian(nn.Module):

    def __init__(self, batch_size, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        '''
        Feedforward hGRU with input layer initialised with gaussian weights 
        (not learnable - no grad), then upsampled to 8 feature maps, and 
        fed to hGRU cell
        '''
        super(FFhGRUwithoutGaussian, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.batch_size=int(batch_size/torch.cuda.device_count())
        self.hgru_size=16
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv00 = nn.Conv3d(3, 1, kernel_size=5, bias=False, padding=2)
        # nn.init.normal_(self.conv00.weight, mean=0.0, std=1.0)
        # self.conv0 = nn.Conv3d(1, 8, kernel_size=7, bias=False, padding=3)
        # part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        # part1 = np.expand_dims(part1, axis=0)
        # print(part1.shape)
        # import pdb; pdb.set_trace()
        # self.conv00.weight.data = torch.FloatTensor(part1)

        # self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        self.unit1 = hConvGRUCell(self.hgru_size, self.hgru_size, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, track_running_stats=False)
        # self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        # init.xavier_normal_(self.conv6.weight)
        # init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        self.fc4 = nn.Linear(1*self.hgru_size*16*16, 1) # the first 2 is for batch size, the second digit is for hGRU dimension
        # self.fc4 = nn.Linear(1*self.hgru_size*8*8, 1) # the first 2 is for batch size, the second digit is for hGRU dimension
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        # z-score normalize input video to the mean and std of the gaussian weight inits
        # x = (x - torch.mean(x, axis=[1, 3, 4], keepdims=True)) / torch.std(x, axis=[1, 3, 4], keepdims=True)
        # average across the RGB channel dimension
        # x=torch.mean(x,axis=1, keepdims=True)
        # import pdb; pdb.set_trace()
        # x=x.permute(0,2,1,3,4)
        # x = (x - torch.mean(x, axis=[2, 3, 4])) / torch.std(x, axis=[2, 3, 4])

        # for i in range(x.shape[0]):
        #     import pdb; pdb.set_trace()
        #     # mu=torch.mean(x[i])
        #     # sigma=torch.std(x[i])
        #     # x[i]=(x[i]-mu)/sigma
        #     x[i] = (x[i] - torch.mean(x[i], axis=[0, 2, 3])) / torch.std(x[i], axis=[0, 2, 3])

        # out=out.permute(0,2,3,4,1)
        # npnp=out.cpu().detach().numpy()
        # np.save("gaussian_responses.npy", npnp)

        # with torch.no_grad():   # stopping the first gausian init input layer from learning 
        out = self.conv00(x)
        # import pdb; pdb.set_trace()
        # out = self.conv0(out) # 1x1 conv to inflate feature maps to 8 dims
        out=out.repeat(1,self.hgru_size,1,1,1)
        # out.requires_grad=True
        out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        internal_state = torch.zeros_like(out[0], requires_grad=False)
        # import pdb;pdb.set_trace()

        for t in range(0,out.shape[0]):
            y=out[t]
            internal_state, g2t = self.unit1(y, internal_state, timestep=t)
            if t == self.timesteps - 2:
                state_2nd_last = internal_state
            elif t == self.timesteps - 1:
                last_state = internal_state        

        # import pdb; pdb.set_trace()
        output = self.bn(internal_state)
        # output = torch.mean(output,1)
        output = self.avgpool(output)
        output=output.view(self.batch_size,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
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




class FFhGRUwithGabor(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(FFhGRUwithGabor, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv00 = nn.Conv3d(3, 25, kernel_size=7, bias=False, padding=3)
        self.conv0 = nn.Conv3d(25, 25, kernel_size=7, bias=False, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        part1 = np.expand_dims(part1, axis=0)
        print(part1.shape)
        self.conv0.weight.data = torch.FloatTensor(part1)

        self.conv1 = nn.Conv3d(1, 25, kernel_size=7, padding=3)

        
        self.unit1 = hConvGRUCell(25, 25, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        self.fc4 = nn.Linear(2*25*64*64, 2) # the first 2 is for batch size
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out = self.conv00(x) # reads the input video
        out = self.conv0(out) # applies gabor weights to the input, gives 1 channel output
        # import pdb; pdb.set_trace()
        out = self.conv1(out) # inflates outputs back to 25 channels to feed to hGRU unit
        out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        internal_state = torch.zeros_like(out[0], requires_grad=False)


        for t in range(0,out.shape[0]):
            y=out[t]
            internal_state, g2t = self.unit1(y, internal_state, timestep=t)
            if t == self.timesteps - 2:
                state_2nd_last = internal_state
            elif t == self.timesteps - 1:
                last_state = internal_state        

        output = self.bn(internal_state)
        # output = torch.mean(output,1)
        # import pdb; pdb.set_trace()
        output = self.avgpool(output)
        output=output.view(1,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, loss

class FFhGRUwithoutGabor(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        
        # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
        self.conv0 = nn.Conv3d(3, 25, kernel_size=7, bias=False, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        # inflate the weight file to accomodate 3 channel, 3D video input
        # part1=np.repeat(part1,3,axis=1)
        part1 = np.expand_dims(part1, axis=0)
        print(part1.shape)
        # self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = hConvGRUCell(25, 25, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

        # self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
        self.fc4 = nn.Linear(2*25*64*64, 2) # the first 2 is for batch size
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        out = self.conv0(x)
        out = torch.pow(out, 2)
        out = out.permute(2,0,1,3,4)
        internal_state = torch.zeros_like(out[0], requires_grad=False)


        for t in range(0,out.shape[0]):
            y=out[t]
            internal_state, g2t = self.unit1(y, internal_state, timestep=t)
            if t == self.timesteps - 2:
                state_2nd_last = internal_state
            elif t == self.timesteps - 1:
                last_state = internal_state        

        output = self.bn(internal_state)
        # output = torch.mean(output,1)
        # import pdb; pdb.set_trace()
        output = self.avgpool(output)
        output=output.view(1,-1)
        output=self.fc4(output)
        output=torch.squeeze(output)
        output=torch.sigmoid(output.clone())
        loss = criterion(output, target.float())


        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.9
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, loss



# class FFhGRU(nn.Module):

#     def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
#         super(FFhGRU, self).__init__()
#         self.timesteps = timesteps
#         self.num_iter = num_iter
#         self.exp_name = exp_name
#         self.jacobian_penalty = jacobian_penalty
#         self.grad_method = grad_method
        
#         # self.conv0 = nn.Conv2d(3, 25, kernel_size=7, padding=3) # using 3 channel input now
#         self.conv0 = nn.Conv3d(3, 25, kernel_size=7, bias=False, padding=3)
#         part1 = np.load('utils/gabor_serre.npy')
#         # inflate the weight file to accomodate 3 channel, 3D video input
#         part1=np.repeat(part1,3,axis=1)
#         part1 = np.expand_dims(part1, axis=0)
#         print(part1.shape)
#         # self.conv0.weight.data = torch.FloatTensor(part1)
        
#         self.unit1 = hConvGRUCell(25, 25, filt_size)
#         print("Training with filter size:", filt_size, "x", filt_size)
#         self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
#         self.conv6 = nn.Conv3d(25, 2, kernel_size=1)
#         init.xavier_normal_(self.conv6.weight)
#         init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

#         self.fc4 = nn.Linear(25*128*128*2, 2) # the first 2 is for batch size
#         self.avgpool = nn.AvgPool3d(kernel_size=1, stride=1)


#     def forward(self, x, epoch, itr, target, criterion, testmode=False):
#         out = self.conv0(x)
#         out = torch.pow(out, 2)
#         out = out.permute(2,0,1,3,4)
#         # internal_states = torch.zeros_like(out, requires_grad=False)
#         internal_state = torch.zeros_like(out[0], requires_grad=False)

#         # import pdb; pdb.set_trace()
#         # xx=x.permute(2,0,1,3,4) # change the shape to have the 64 frames first
#         outputs = []

#         for t in range(0,out.shape[0]):
#             y=out[t]

#             # x = self.conv0(y)
#             # x = torch.pow(x, 2)
#             # internal_state = torch.zeros_like(x, requires_grad=False)
#             # internal_state = internal_states[t]
#             # import pdb; pdb.set_trace()
#             states = []
#             if self.grad_method == 'rbp':
#                 with torch.no_grad():
#                     for i in range(self.timesteps - 1):
#                         if testmode: states.append(internal_state)
#                         internal_state, g2t = self.unit1(y, internal_state, timestep=i)
#                 if testmode: states.append(internal_state)
#                 state_2nd_last = internal_state.detach().requires_grad_()
#                 i += 1
#                 last_state, g2t = self.unit1(y, state_2nd_last, timestep=i)
#                 internal_state = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
#                 if testmode: states.append(internal_state)

#             elif self.grad_method == 'bptt':
#                 for i in range(self.timesteps):
#                     internal_state, g2t = self.unit1(y, internal_state, timestep=i)
#                     if i == self.timesteps - 2:
#                         state_2nd_last = internal_state
#                     elif i == self.timesteps - 1:
#                         last_state = internal_state
#             #internal_state = torch.tanh(internal_state)
#             output = self.bn(internal_state)
#             outputs.append(output)
#             # import pdb; pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         output = torch.stack(outputs)
#         # output = output.permute(1,2,0,3,4)
#         # output = self.conv6(output)
#         # global average pooling
#         output = torch.mean(output,0)
#         output = self.avgpool(output)
#         # import pdb; pdb.set_trace()
#         output=output.view(1,-1)
#         output=self.fc4(output)
#         output=torch.squeeze(output)
#         output=torch.sigmoid(output.clone())
#         # import pdb; pdb.set_trace()
#         # loss = criterion(torch.sigmoid(output, inplace=False), target.float())
#         loss = criterion(output, target.float())


#         pen_type = 'l1'
#         jv_penalty = torch.tensor([1]).float().cuda()
#         mu = 0.9
#         double_neg = False
#         if self.training and self.jacobian_penalty:
#             if pen_type == 'l1':
#                 norm_1_vect = torch.ones_like(last_state)
#                 norm_1_vect.requires_grad = False
#                 jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
#                                               create_graph=self.jacobian_penalty, allow_unused=True)[0]
#                 jv_penalty = (jv_prod - mu).clamp(0) ** 2
#                 if double_neg is True:
#                     neg_norm_1_vect = -1 * norm_1_vect.clone()
#                     jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
#                                                   create_graph=True, allow_unused=True)[0]
#                     jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
#                     jv_penalty = jv_penalty + jv_penalty2
#             elif pen_type == 'idloss':
#                 norm_1_vect = torch.rand_like(last_state).requires_grad_()
#                 jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
#                                               create_graph=True, allow_unused=True)[0]
#                 jv_penalty = (jv_prod - norm_1_vect) ** 2
#                 jv_penalty = jv_penalty.mean()
#                 if torch.isnan(jv_penalty).sum() > 0:
#                     raise ValueError('Nan encountered in penalty')
#         if testmode: return output, states, loss
#         return output, jv_penalty, loss


class hConvGRUallSUP(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(hConvGRUallSUP, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = hConvGRUCell(25, 25, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        x = self.conv0(x)
        x = torch.pow(x, 2)
        internal_state = torch.zeros_like(x, requires_grad=False)
        states = []
        if self.grad_method == 'rbp':
            with torch.no_grad():
                for i in range(self.timesteps - 1):
                    if testmode: states.append(internal_state)
                    internal_state, g2t = self.unit1(x, internal_state, timestep=i)
            if testmode: states.append(internal_state)
            state_2nd_last = internal_state.detach().requires_grad_()
            i += 1
            last_state, g2t = self.unit1(x, state_2nd_last, timestep=i)
            internal_state = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
            if testmode: states.append(internal_state)

        elif self.grad_method == 'bptt':
            for i in range(self.timesteps):
                internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                if i == self.timesteps - 2:
                    state_2nd_last = internal_state
                elif i == self.timesteps - 1:
                    last_state = internal_state
        
                output = self.bn(internal_state)
                output = self.conv6(output)
                loss = criterion(output, target)
                if i == 0:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.90
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, total_loss


class hConvGRUtrunc(nn.Module):

    def __init__(self, timesteps=8, filt_size=15, num_iter=50, exp_name='exp1', jacobian_penalty=False, grad_method='bptt'):
        super(hConvGRUtrunc, self).__init__()
        self.timesteps = timesteps
        self.num_iter = num_iter
        self.exp_name = exp_name
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        
        self.conv0 = nn.Conv2d(1, 25, kernel_size=7, padding=3)
        part1 = np.load('utils/gabor_serre.npy')
        self.conv0.weight.data = torch.FloatTensor(part1)
        
        self.unit1 = hConvGRUCell(25, 25, filt_size)
        print("Training with filter size:", filt_size, "x", filt_size)
        self.bn = nn.BatchNorm2d(25, eps=1e-03, track_running_stats=False)
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        init.xavier_normal_(self.conv6.weight)
        init.constant_(self.conv6.bias, torch.log(torch.tensor((1 - 0.01) / 0.01)))

    def forward(self, x, epoch, itr, target, criterion, testmode=False):
        x = self.conv0(x)
        x = torch.pow(x, 2)
        internal_state = torch.zeros_like(x, requires_grad=False)
        states = []
        
        tk = 3

        if self.grad_method == 'rbp':
            with torch.no_grad():
                for i in range(self.timesteps - 1):
                    if testmode: states.append(internal_state)
                    internal_state, g2t = self.unit1(x, internal_state, timestep=i)
            if testmode: states.append(internal_state)
            state_2nd_last = internal_state.detach().requires_grad_()
            i += 1
            last_state, g2t = self.unit1(x, state_2nd_last, timestep=i)
            internal_state = dummyhgru.apply(state_2nd_last, last_state, epoch, itr, self.exp_name, self.num_iter)
            if testmode: states.append(internal_state)


        elif self.grad_method == 'bptt':
            for i in range(self.timesteps):
                if i < 3:
                    with torch.no_grad():
                        internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                else:
                    internal_state, g2t = self.unit1(x, internal_state, timestep=i)
                if i == self.timesteps - 2:
                    state_2nd_last = internal_state
                elif i == self.timesteps - 1:
                    last_state = internal_state
        
        output = self.bn(internal_state)
        output = self.conv6(output)
        loss = criterion(output, target)

        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        mu = 0.90
        double_neg = False
        if self.training and self.jacobian_penalty:
            if pen_type == 'l1':
                norm_1_vect = torch.ones_like(last_state)
                norm_1_vect.requires_grad = False
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=self.jacobian_penalty, allow_unused=True)[0]
                jv_penalty = (jv_prod - mu).clamp(0) ** 2
                if double_neg is True:
                    neg_norm_1_vect = -1 * norm_1_vect.clone()
                    jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[neg_norm_1_vect], retain_graph=True,
                                                  create_graph=True, allow_unused=True)[0]
                    jv_penalty2 = (jv_prod - mu).clamp(0) ** 2
                    jv_penalty = jv_penalty + jv_penalty2
            elif pen_type == 'idloss':
                norm_1_vect = torch.rand_like(last_state).requires_grad_()
                jv_prod = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=[norm_1_vect], retain_graph=True,
                                              create_graph=True, allow_unused=True)[0]
                jv_penalty = (jv_prod - norm_1_vect) ** 2
                jv_penalty = jv_penalty.mean()
                if torch.isnan(jv_penalty).sum() > 0:
                    raise ValueError('Nan encountered in penalty')
        if testmode: return output, states, loss
        return output, jv_penalty, loss
