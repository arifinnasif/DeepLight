import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.conv_lstm import *
from layers.mb_lstm import *
from collections import OrderedDict




class StepDeep(nn.Module):
    def __init__(self, future_hours, feature_count):
        super(StepDeep, self).__init__()
        self.num_frames_truth = 1
        self.num_frames = 6
        self.fea_dim = 1

        self.conv1 = nn.Sequential( 
            nn.Conv3d(self.num_frames_truth, 128, kernel_size=(3,1,1), stride=1, padding=(1,0,0)),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(5,1,1), stride=1, padding=(2,0,0)),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3,3,3), stride=1, padding=(1,1,1)),
            nn.ReLU()
        )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(7,7), stride=1, padding=(3,3)),
            nn.ReLU()
        )

        self.conv2d_2 = nn.Conv2d(64, 1, kernel_size=(7,7), stride=1, padding=(3,3))
        

    
    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:1,:,:]
        input_batch = input_batch.permute(0, 2, 1, 3, 4)
        result = []
        
        output = self.conv1(input_batch)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        
        output = output.permute(0, 2, 1, 3, 4)
        output = output.reshape(-1, output.shape[2], 159, 159)
        output = self.conv2d_1(output)
        output = self.conv2d_2(output)
        output = output.reshape(-1, 6, output.shape[1], 159, 159)

        return output
        
        for i in range(6):
            x = output[:, :, i, :, :]
            x = self.conv2d_1(x)
            x = self.conv2d_2(x)
            result.append(x)
        
        result = torch.stack(result, dim=2)
        return F.sigmoid(result.permute(0, 2, 1, 3, 4))
        


class LightNet_O(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels):
        super(LightNet_O, self).__init__()
        self.obs_tra_frames = obs_tra_frames
        self.future_frames = 6
        self.obs_channels=obs_channels
        
        self.obs_encoder_module = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ) # (bs, 4, 80, 80)

        self.encoder_ConvLSTM = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=80)

        self.encoder_h = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.encoder_c = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
        ) # (bs, 4, 80, 80)

        self.decoder_ConvLSTM = ConvLSTM2D(4, 64, kernel_size=5, img_rowcol=80) # first on is the output channels channels of decoder_1 and second one is the hidden channels

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1),
            
        )


    def forward(self, obs):
        batch_size = obs.shape[0]

        obs=obs[:,:,0:1]


        h = torch.zeros([batch_size, 8, 80, 80], dtype=torch.float32).to(obs.device)
        c = torch.zeros([batch_size, 8, 80, 80], dtype=torch.float32).to(obs.device)


        for t in range(self.obs_tra_frames):
            obs_encoder = self.obs_encoder_module(obs[:,t,0:1]) # (bs, 4, 80, 80)
            h, c = self.encoder_ConvLSTM(obs_encoder, h, c) # (bs, 8, 80, 80), (bs, 8, 80, 80)
        h = self.encoder_h(h) # (bs, 64, 80, 80)
        c = self.encoder_c(c) # (bs, 64, 80, 80)


        last_frame = obs[:, -1, 0:1]

        out_list = []

        for t in range(self.future_frames):
            x = self.decoder_1(last_frame) # (bs, 4, 80, 80)
            h, c = self.decoder_ConvLSTM(x, h, c) # (bs, 8, 80, 80), (bs, 8, 80, 80)
            x =  self.decoder_2(c) # (bs, 1, 159, 159)
            out_list.append(x)
            last_frame = F.sigmoid(x)

        return torch.cat(out_list, dim=1).unsqueeze(2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ADSNet_O(nn.Module):
    def __init__(self, obs_tra_frames, obs_channels):
        super(ADSNet_O, self).__init__()
        self.num_frames_truth = obs_channels

        # Encoder
        self.encoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(self.num_frames_truth, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        ) # output shape: (batch_size, 4, 80, 80)

        self.encoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        ) # output shape: (batch_size, 8, 40, 40)

        self.en_convlstm = ConvLSTM2D(4, 8, kernel_size=5, img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)
        )

        self.decoder_conv2d_2 = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        )

        # self.de_convlstm = ConvLSTMCell(4, 16, kernel_size=(5, 5))
        self.de_convlstm = ConvLSTM2D(4, 16, kernel_size=5, img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 80, 80)

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) # output shape: (batch_size, 32, 160, 160)

        self.de_conv_out = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Encoder
        for t in range(6):

            x = self.encoder_conv2d_1(input_batch[:, t, :, :, :])
            x = self.encoder_conv2d_2(x)
            if t == 0:
                h, c = torch.zeros([batch_size, 8, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 8, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                h, c = self.en_convlstm(x, h, c)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)



class HazyLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.BCELoss = nn.BCELoss(reduction='none')

  def forward(self, prediction, truth, transformed_ground_truth):
    # as prediction is not sigmoided
    prediction = torch.sigmoid(prediction)
    weight = (1-transformed_ground_truth)*prediction + transformed_ground_truth*(1-prediction)
    loss = self.BCELoss(prediction, truth)
    loss = loss * weight
    return torch.mean(loss)
  

    
class CStemMultiBranchConvSubBlock(nn.Module):
    def __init__(self, input_frames, output_frames, kernel, max_pool_padding):
        super().__init__()
        self.conv = nn.Conv2d(input_frames, output_frames, kernel_size=kernel, padding=kernel//2)
        self.bn = nn.BatchNorm2d(output_frames)
        self.relu = nn.ReLU()
        # self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=max_pool_padding)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.max_pool(x)
        return x
    

class CStemMultiBranchConvBlock(nn.Module):
    def __init__(self, input_frames, output_frames, list_of_kernels, max_pool_padding):
        super().__init__()
        self.list_of_kernels = list_of_kernels
        self.sub_blocks = nn.ModuleList()
        self.permutation = nn.Conv2d(output_frames, output_frames, kernel_size=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=max_pool_padding)
        for kernel in list_of_kernels:
            self.sub_blocks.append(CStemMultiBranchConvSubBlock(input_frames, output_frames//len(list_of_kernels), kernel, max_pool_padding))
        
    
    def forward(self, x):
        out = []
        for sub_block in self.sub_blocks:
            out.append(sub_block(x))
        # print(torch.cat(out, dim=1).shape)
        out = self.permutation(torch.cat(out, dim=1))
        # print(out.shape)
        out = F.relu(out)
        out = self.max_pool(out)
        # print(out.shape)
        return out
        



class DeepLight(nn.Module):
    """multipath model dual encoder"""
    def __init__(self, obs_tra_frames, obs_channels):
        super(DeepLight, self).__init__()
        self.num_frames_truth = obs_channels

        # Lightning Encoder
        self.lightning_encoder_conv2d_1 = CStemMultiBranchConvBlock(3, 32, [5,11,21,31], 1) 

        self.lightning_encoder_conv2d_2 = nn.Sequential(
            CStemMultiBranchConvBlock(32, 64, [1,5,11,15], 0),
            nn.LayerNorm([64, 40, 40], elementwise_affine=True)
        ) 

        

        self.lightning_en_convlstm = MBConvLSTM(64, 64, [3,5,7,11], img_rowcol=40)

        # Auxiliary Encoder
        self.other_encoder_conv2d_1 = CStemMultiBranchConvBlock(self.num_frames_truth-3, 32, [5,11,21,31], 1)

        self.other_encoder_conv2d_2 = nn.Sequential(
            CStemMultiBranchConvBlock(32, 64, [1,5,11,15], 0),
            nn.LayerNorm([64, 40, 40], elementwise_affine=True)
        ) 

        self.other_en_convlstm = MBConvLSTM(64, 64, [3,5,7,11], img_rowcol=40)

        self.en_de_h = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        self.en_de_c = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
            nn.ReLU()
        )

        # Decoder

        self.decoder_conv2d_1 = CStemMultiBranchConvBlock(1, 16, [5,11,21,31], 1)

        self.decoder_conv2d_2 = nn.Sequential(
            CStemMultiBranchConvBlock(16, 32, [1,5,11,15], 0),
            nn.LayerNorm([32, 40, 40], elementwise_affine=True)
        )

        
        self.de_convlstm = MBConvLSTM(32, 128, [3,5,7,11], img_rowcol=40)

        self.de_conv2dT_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) 

        self.de_conv2dT_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1),
            nn.ReLU()
        ) 

        self.de_conv_out = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0)



        

    def forward(self, input_batch):
        input_batch = input_batch[:,:,0:self.num_frames_truth,:,:]
        last_frame = input_batch[:, -1, 0:1, :, :]
        batch_size = input_batch.shape[0]

        
        # Lightning Encoder
        for t in range(6):
            x = self.lightning_encoder_conv2d_1(input_batch[:, t, 0:3, :, :])
            # print(x.shape)
            x = self.lightning_encoder_conv2d_2(x)
            if t == 0:
                lightning_h, lightning_c = torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                lightning_h, lightning_c = self.lightning_en_convlstm(x, lightning_h, lightning_c)

        # Other Encoder
        for t in range(6):
            x = self.other_encoder_conv2d_1(input_batch[:, t, 3:, :, :])
            x = self.other_encoder_conv2d_2(x)
            if t == 0:
                other_h, other_c = torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device), torch.zeros([batch_size, 64, 40, 40], dtype=torch.float32).to(input_batch.device)
            else:
                other_h, other_c = self.other_en_convlstm(x, other_h, other_c)

        h = torch.cat([lightning_h, other_h], dim=1)
        c = torch.cat([lightning_c, other_c], dim=1)
        
        del x
        del input_batch

        # Encoder to Decoder
        h = self.en_de_h(h)
        c = self.en_de_c(c)

        # decoder

        out_list = []

        for t in range(6):
            x = self.decoder_conv2d_1(last_frame)
            x = self.decoder_conv2d_2(x)
            h, c = self.de_convlstm(x, h, c)
            x = self.de_conv2dT_1(c)
            x = self.de_conv2dT_2(x)
            x = self.de_conv_out(x)
            x = x[:,:,:-1,:-1]
            out_list.append(x)
            last_frame = F.sigmoid(x)


        return torch.cat(out_list, dim=1).unsqueeze(2)
    





# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias= True)

class _bn_relu_conv(nn.Module):
    def __init__(self, nb_filter, bn = False):
        super(_bn_relu_conv, self).__init__()
        self.has_bn = bn
        #self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = torch.relu
        self.conv1 = conv3x3(nb_filter, nb_filter)

    def forward(self, x):
        #if self.has_bn:
        #    x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x

class _residual_unit(nn.Module):
    def __init__(self, nb_filter, bn=False):
        super(_residual_unit, self).__init__()
        self.bn_relu_conv1 = _bn_relu_conv(nb_filter, bn)
        self.bn_relu_conv2 = _bn_relu_conv(nb_filter, bn)

    def forward(self, x):
        residual = x

        out = self.bn_relu_conv1(x)
        out = self.bn_relu_conv2(out)

        out += residual # short cut

        return out

class ResUnits(nn.Module):
    def __init__(self, residual_unit, nb_filter, repetations=1):
        super(ResUnits, self).__init__()
        self.stacked_resunits = self.make_stack_resunits(residual_unit, nb_filter, repetations)

    def make_stack_resunits(self, residual_unit, nb_filter, repetations):
        layers = []

        for i in range(repetations):
            layers.append(residual_unit(nb_filter))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stacked_resunits(x)

        return x

# Matrix-based fusion
class TrainableEltwiseLayer(nn.Module):
    def __init__(self, n, h, w):
        super(TrainableEltwiseLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, n, h, w),
                                    requires_grad = True)  # define the trainable parameter

    def forward(self, x):
        # assuming x is of size b-1-h-w
        # print('x:', x.shape)
        # print('weights:', self.weights.shape)
        x = x * self.weights # element-wise multiplication

        return x


class stresnet_modified(nn.Module):
    def __init__(self, dummy1, dummy2):
        '''
            C - Temporal Closeness
            P - Period
            T - Trend
            conf = (len_seq, nb_flow, map_height, map_width)
            external_dim
        '''

        super(stresnet_modified, self).__init__()
        # logger = logging.getLogger(__name__)
        # logger.info('initializing net params and ops ...')
        c_conf=None
        p_conf=None
        t_conf=(6, 1, 159, 159)
        external_dim=0
        nb_residual_unit=3

        self.external_dim = external_dim
        self.nb_residual_unit = nb_residual_unit
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf

        self.nb_flow, self.map_height, self.map_width = t_conf[1], t_conf[2], t_conf[3]

        self.relu = torch.relu
        self.tanh = torch.tanh
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.qr_nums = len(self.quantiles)

        if self.c_conf is not None:
            self.c_way = self.make_one_way(in_channels = self.c_conf[0] * self.nb_flow)

        # Branch p
        if self.p_conf is not None:
            self.p_way = self.make_one_way(in_channels = self.p_conf[0] * self.nb_flow)

        # Branch t
        self.t_way = nn.ModuleList()
        if self.t_conf is not None:
            # temp = self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow)
            self.t_way = nn.ModuleList([self.make_one_way(in_channels = self.t_conf[0] * self.nb_flow) for _ in range(6)])

        # Operations of external component
        if self.external_dim != None and self.external_dim > 0:
            self.external_ops = nn.Sequential(OrderedDict([
            ('embd', nn.Linear(self.external_dim, 10, bias = True)),
                ('relu1', nn.ReLU()),
            ('fc', nn.Linear(10, self.nb_flow * self.map_height * self.map_width, bias = True)),
                ('relu2', nn.ReLU()),
            ]))

    def make_one_way(self, in_channels):

        return nn.Sequential(OrderedDict([
            ('conv1', conv3x3(in_channels = in_channels, out_channels = 64)),
            ('ResUnits', ResUnits(_residual_unit, nb_filter = 64, repetations = self.nb_residual_unit)),
            ('relu', nn.ReLU()),
            ('conv2', conv3x3(in_channels = 64, out_channels = self.nb_flow)),
            ('FusionLayer', TrainableEltwiseLayer(n = self.nb_flow, h = self.map_height, w = self.map_width))
        ]))

    def forward(self, input_t):
        input_c = None
        input_p = None
        input_ext = None

        # Three-way Convolution
        main_output = 0
        if self.c_conf is not None:
            input_c = input_c.view(-1, self.c_conf[0]*self.nb_flow, self.map_height, self.map_width)
            out_c = self.c_way(input_c)
            main_output += out_c
        if self.p_conf is not None:
            input_p = input_p.view(-1, self.p_conf[0]*self.nb_flow, self.map_height, self.map_width)
            out_p = self.p_way(input_p)
            main_output += out_p
        if self.t_conf is not None:
            input_t = input_t.view(-1, self.t_conf[0]*self.nb_flow, self.map_height, self.map_width)
            # print("hello")
            # print(input_t.shape)
            # print("hello")
            out_t = []
            for i in range(6):
                out_t.append(self.t_way[i](input_t))
            
            out_t = torch.stack(out_t, dim=1)
            # print("hello")
            # print(out_t.shape)
            
            main_output += out_t
            main_output += out_t
            # print(main_output.shape)
            # print("bye")
            # exit(-1)

        # parameter-matrix-based fusion
        #main_output = out_c + out_p + out_t

        # fusing with external component
        if self.external_dim != None and self.external_dim > 0:
            # external input
            external_output = self.external_ops(input_ext)
            external_output = self.relu(external_output)
            external_output = external_output.view(-1, self.nb_flow, self.map_height, self.map_width)
            #main_output = torch.add(main_output, external_output)
            main_output += external_output

        else:
            # print('external_dim:', self.external_dim)
            pass


        # main_output = self.tanh(main_output)

        return main_output


class InceptionFromGoogLeNet(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(InceptionFromGoogLeNet, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
