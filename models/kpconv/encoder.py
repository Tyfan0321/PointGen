import torch
import torch.nn as nn

from models.kpconv.modules import ConvBlock, ResidualBlock


class KPConvEncoder(nn.Module):
    def __init__(
        self, 
        kpconv_layers,
        input_dim,
        kernel_size,
        init_dim,
        init_sigma,
        init_radius,
    ):
        super(KPConvEncoder, self).__init__()
        self.kpconv_layers = kpconv_layers
        self.input_dim = input_dim
        self.init_dim = init_dim
        self.kernel_size = kernel_size
        inter_dim = self.init_dim * 2
        inter_sigma = init_sigma
        inter_radius = init_radius

        self.encoder1_1 = ConvBlock(self.input_dim, self.init_dim, self.kernel_size, inter_radius, inter_sigma)
        self.encoder1_2 = ResidualBlock(self.init_dim, inter_dim, self.kernel_size, inter_radius, inter_sigma)
        
        self.encoder = nn.ModuleList()
        for _ in range(1, self.kpconv_layers):
            self.encoder.append(nn.ModuleList([
                ResidualBlock(inter_dim, inter_dim, self.kernel_size, inter_radius, inter_sigma, strided=True),
                ResidualBlock(inter_dim, inter_dim * 2, self.kernel_size, inter_radius * 2, inter_sigma * 2),
                ResidualBlock(inter_dim * 2, inter_dim * 2, self.kernel_size, inter_radius * 2, inter_sigma * 2),
            ]))
            inter_dim = inter_dim * 2
            inter_sigma = inter_sigma * 2
            inter_radius = inter_radius * 2
    
    def forward(self, points_list, neighbors_list, subsampling_list):
        feats = torch.ones_like(points_list[0][:, :1])
        feats = self.encoder1_1(feats, points_list[0], points_list[0], neighbors_list[0])
        feats = self.encoder1_2(feats, points_list[0], points_list[0], neighbors_list[0])

        feats_list = [feats]
        for i in range(self.kpconv_layers - 1):
            feats = self.encoder[i][0](feats, points_list[i + 1], points_list[i], subsampling_list[i])
            feats = self.encoder[i][1](feats, points_list[i + 1], points_list[i + 1], neighbors_list[i + 1])
            feats = self.encoder[i][2](feats, points_list[i + 1], points_list[i + 1], neighbors_list[i + 1])
            feats_list.append(feats)
        
        return feats_list
