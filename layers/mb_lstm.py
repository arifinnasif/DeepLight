# from github.com/andyflying/lightnet_plus
import torch
import torch.nn as nn
import torch.nn.functional as F


    

class LSTMMultiBranchConvSubBlock(nn.Module):
    def __init__(self, input_frames, output_frames, kernel, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(input_frames, output_frames, kernel_size=kernel, padding=kernel//2, bias=bias)
        # self.bn = nn.BatchNorm2d(output_frames)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.relu(x)
        
        return x
    

class LSTMMultiBranchConvBlock(nn.Module):
    def __init__(self, input_frames, output_frames, list_of_kernels, bias):
        super().__init__()
        self.list_of_kernels = list_of_kernels
        self.sub_blocks = nn.ModuleList()
        self.permutation = nn.Conv2d(output_frames, output_frames, kernel_size=1, padding=0, bias=bias)
        for kernel in list_of_kernels:
            self.sub_blocks.append(LSTMMultiBranchConvSubBlock(input_frames, output_frames//len(list_of_kernels), kernel))
        
    
    def forward(self, x):
        out = []
        for sub_block in self.sub_blocks:
            out.append(sub_block(x))
        out = self.permutation(torch.cat(out, dim=1))
        # out = F.relu(out)
        
        return out

  
class MBConvLSTM(nn.Module):
    def __init__(self, channels, filters, list_of_kernels, img_rowcol):
        super().__init__()
        # self.channels = channels
        self.filters = filters
        # self.padding = kernel_size // 2
        # self.kernel_size = kernel_size
        # self.strides = strides
        # self.conv_x = nn.Conv2d(channels, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)
        self.conv_x = LSTMMultiBranchConvBlock(channels, filters*4, list_of_kernels, bias=True)
        # self.conv_h = nn.Conv2d(filters, filters * 4, kernel_size=kernel_size, stride=1, padding=self.padding, bias=False)
        self.conv_h = LSTMMultiBranchConvBlock(filters, filters*4, list_of_kernels, bias=False)
        self.mul_c = nn.Parameter(torch.zeros([1, filters * 3, img_rowcol, img_rowcol], dtype=torch.float32))


    def forward(self, x, h, c):
        # x -> [batch_size, channels, x, y]
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        i_x, f_x, c_x, o_x = torch.split(x_concat, self.filters, dim=1)
        i_h, f_h, c_h, o_h = torch.split(h_concat, self.filters, dim=1)
        i_c, f_c, o_c = torch.split(self.mul_c, self.filters, dim=1)
        i_t = torch.sigmoid(i_x + i_h + i_c * c)
        f_t = torch.sigmoid(f_x + f_h + f_c * c)
        c_t = torch.tanh(c_x + c_h)
        c_next = i_t * c_t + f_t * c
        o_t = torch.sigmoid(o_x + o_h + o_c * c_next)
        h_next = o_t * torch.tanh(c_next)
        return h_next, c_next
    
