from model import common
# from model.deformable_kernels.deformable_kernels.modules import DeformKernelConv2d
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np 

url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

  
class CGConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CGConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if kernel_size == 1 or True:
            self.ind = True
        else:
            self.ind = False            
            self.oc = out_channels
            self.ks = kernel_size
            
            # the target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws,ws))
            
            # the dimension of the latent repsentation
            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)
            
            # the context encoding module
            self.ce = nn.Linear(ws*ws, num_lat, False)            
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)
            
            # activation function is relu
            self.act = nn.ReLU(inplace=True)
            
            
            # the number of groups in the channel interacting module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            # the channel interacting module    
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)
            
            # the gate decoding module
            self.gd = nn.Linear(num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(num_lat, kernel_size * kernel_size, False)
            
            # used to prrepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
            
            # sigmoid function
            self.sig = nn.Sigmoid()
    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight
            # allocate glbal information
            gl = self.avg_pool(x).view(b,c,-1)
            # context-encoding module
            out = self.ce(gl)
            # use different bn for the following two branches
            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)
            # gate decoding branch 1
            out = self.gd(out)
            # channel interacting module
            if self.g >3:
                # grouped linear
                oc = self.ci(self.act(self.ci_bn2(ce2).\
                                      view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:
                # linear layer for resnet.conv1
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous() 
            oc = oc.view(b,self.oc,-1) 
            oc = self.ci_bn(oc)
            oc = self.act(oc)
            # gate decoding branch 2
            oc = self.gd2(oc)   
            # produce gate
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))
            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()
            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)
            # currently only handle square input and output
            return torch.matmul(out,x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))   
            

def gated_conv(in_channels, out_channels, kernel_size, bias=True):
    return CGConv2d(in_channels, out_channels, kernel_size=kernel_size, 
                    padding=(kernel_size//2), stride=1, bias=bias)
        

def make_model(args, parent=False):
    return CGC_EDSR(args)


class CGC_EDSR(nn.Module):
    def __init__(self, args, conv=common.default_conv, gated_conv=gated_conv):
        super(CGC_EDSR, self).__init__()
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'cgc_edsr_r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = []
        if not args.one_cgc_block:
            m_body = [
                common.ResBlock(
                    gated_conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
                ) for _ in range(n_resblocks)
            ]
            m_body.append(gated_conv(n_feats, n_feats, kernel_size))
        else:
            for i in range(n_resblocks-1):
                m_body.append(common.ResBlock(conv, n_feats, kernel_size, 
                                              act=act, res_scale=args.res_scale))
            
            m_body.append(common.ResBlock(gated_conv, n_feats, kernel_size, act=act, res_scale=args.res_scale))
            m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

