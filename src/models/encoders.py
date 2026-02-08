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
    
    def forward(self, points_list, neighbors_list, subsampling_list):
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
            assert "pretrained_ckpt" is None, "Only support pretrained model"
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        
        base_channels = self.model.enc_channels[-1]
        self.out_channels = base_channels * 3  

        # default transform pipeline
        from src.models.sonata import transform
        self.transform = transform.default()
    
    def forward(self, point):
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].numpy()
        point = self.transform(point)

        with torch.inference_mode():
            for key in point.keys():
                if isinstance(point[key], torch.Tensor):
                    point[key] = torch.from_numpy(point[key]).cuda(non_blocking=True)
            
            point = self.model(point)
            
            for _ in range(2):
                if "pooling_parent" not in point:
                    break
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            
            while "pooling_parent" in point:
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = point.feat[inverse]
                point = parent
            
            feats = point.feat[point.inverse]
        
        return feats

