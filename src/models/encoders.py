import torch


class KPConvEncoder(torch.nn.Module):
    """KPConv encoder for point cloud feature extraction."""
    
    def __init__(
        self,
        kpconv_layers=4,
        input_dim=1,
        kernel_size=15,
        init_dim=64,
        init_sigma=0.05,
        init_radius=0.0625,
        **kwargs
    ):
        super().__init__()
        
        from src.models.kpconv.encoder import KPConv
        self.model = KPConv(
            kpconv_layers=kpconv_layers,
            input_dim=input_dim,
            kernel_size=kernel_size,
            init_dim=init_dim,
            init_sigma=init_sigma,
            init_radius=init_radius
        )
        
        hidden_dim = init_dim * 2 * (2 ** (kpconv_layers - 1))
        self.out_channels = hidden_dim
    
    def forward(self, points_list, neighbors_list, subsampling_list, length_list):
        feats_list= self.model(points_list, neighbors_list, subsampling_list)
        feats_c = feats_list[-1]

        ref_points_c = points_list[-1][:length_list[-1][0]]
        src_points_c = points_list[-1][length_list[-1][0]:]
        assert feats_c.shape[0] == ref_points_c.shape[0] + src_points_c.shape[0]
        ref_feats, src_feats = torch.split(feats_c, [ref_points_c.shape[0], src_points_c.shape[0]], dim=0)

        return ref_feats, src_feats


class SonataEncoder(torch.nn.Module):
    """Sonata encoder for point cloud feature extraction."""
    
    def __init__(
        self,
        pretrained=True,
        freeze=True,
        pretrained_ckpt=None,
        layer_index=0,
        **kwargs
    ):
        super().__init__()
        
        if pretrained:
            from src.models.sonata.model import load
            if pretrained_ckpt:
                self.model = load(pretrained_ckpt)
            else:
                self.model = load("sonata")
        else:
            assert not pretrained_ckpt, "Only support pretrained model"
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.enc_channels = list(self.model.enc_channels)
        self.out_channels = self.enc_channels[layer_index]

        # default transform pipeline
        from src.models.sonata import transform
        self.transform = transform.default()
    
    def forward(self, point):
        assert "feat" in point, "point needs to be processed"
        with torch.inference_mode():
            point = self.model(point)

            layers = []
            current = point
            while True:
                coord = current.coord if "coord" in current else current.get("origin_coord", None)
                layers.append({"coord": coord, "feat": current.feat})
                if "pooling_parent" not in current:
                    break
                current = current.pooling_parent

            # for _ in range(2):
            #     if "pooling_parent" not in point:
            #         break
            #     parent = point.pop("pooling_parent")
            #     inverse = point.pop("pooling_inverse")
            #     parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            #     point = parent
            
            # while "pooling_parent" in point:
            #     parent = point.pop("pooling_parent")
            #     inverse = point.pop("pooling_inverse")
            #     parent.feat = point.feat[inverse]
            #     point = parent
            
            # _ = point.feat[point.inverse]   
        return layers
