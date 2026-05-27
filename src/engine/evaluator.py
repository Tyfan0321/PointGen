import torch
from tqdm import tqdm
from src.models.pipeline import PointGenPipeline
from src.utils.point_cloud_utils import weighted_svd

from src.models.pipeline import Evaluator

class DiffusionEvaluator:    
    def __init__(self, cfg, processor=None, noise_scheduler=None):
        self.evaluator = Evaluator(
            rre_threshold=cfg.eval.rre_threshold,
            rte_threshold=cfg.eval.rte_threshold,
        )
        self.num_gen_samples = cfg.num_gen_samples
        self.num_inference_steps = cfg.num_inference_steps
        self.inference_type = cfg.inference_type
        self.use_overlap_metrics = cfg.eval.get("use_overlap_metrics", False)
        self.overlap_radius = cfg.data.voxel_size * 1.5
        self.processor = processor
        self.noise_scheduler = noise_scheduler
    
    def evaluate(self, model, dataloader):
        transformer = model
        pipeline = PointGenPipeline(
            scheduler=self.noise_scheduler,
            processor=self.processor,
            transformer=transformer,
            scheduler_type=self.inference_type,
        )
        generator = torch.Generator(device=model.device).manual_seed(0)
        
        dist_error = []
        dist_ov_error = []
        all_re = []
        all_te = []
        all_rr = []
        all_re_ov = []
        all_te_ov = []
        all_rr_ov = []
        all_re_w = []
        all_te_w = []
        all_rr_w = []
        overlap_ratios = []
        
        progress_bar = tqdm(dataloader, desc="Evaluation")
        
        for i, data_dict in enumerate(progress_bar):
            pred_points, tgt_points, ref_points, tgt_points_corr, src_points, gt_overlap = pipeline(
                data_dict, 
                num_inference_steps=self.num_inference_steps, 
                generator=generator
            )
            
            pred_points = pred_points.squeeze(0)
            pred_points = pred_points * torch.std(ref_points) + torch.mean(ref_points, dim=0)
            
            per_point_dist_error = torch.norm(pred_points - tgt_points, dim=-1).mean()
            dist_error.append(per_point_dist_error)
            
            pred_transform = weighted_svd(src_points.squeeze(0), pred_points)
            te, re, rr = self.compute_transform_error(data_dict["Tr"].squeeze(0), pred_transform)
            all_te.append(te.float().item())
            all_re.append(re.float().item())
            all_rr.append(rr.float().item())

            if self.use_overlap_metrics:
                dist_to_ref = torch.cdist(tgt_points, ref_points).min(dim=1).values
                overlap_weights = (dist_to_ref <= self.overlap_radius).float()
                overlap_mask = overlap_weights > 0.5
                overlap_ratios.append(overlap_mask.float().mean().item())

                if overlap_weights.sum() > 0:
                    pred_transform_w = weighted_svd(
                        src_points.squeeze(0),
                        pred_points,
                        weights=overlap_weights,
                    )
                    te_w, re_w, rr_w = self.compute_transform_error(data_dict["Tr"].squeeze(0), pred_transform_w)
                    all_re_w.append(re_w.float().item())
                    all_te_w.append(te_w.float().item())
                    all_rr_w.append(rr_w.float().item())

                if gt_overlap is not None and gt_overlap.numel() == pred_points.shape[0]:
                    overlap_weights = gt_overlap.float().reshape(-1).to(pred_points.device)
                    overlap_mask = overlap_weights > 0.5

                if overlap_mask.sum() >= 3:
                    ov_dist_error = torch.norm(pred_points[overlap_mask] - tgt_points[overlap_mask], dim=-1).mean()
                    dist_ov_error.append(ov_dist_error)
                    pred_transform_ov = weighted_svd(src_points.squeeze(0)[overlap_mask], pred_points[overlap_mask])
                    te_ov, re_ov, rr_ov = self.compute_transform_error(data_dict["Tr"].squeeze(0), pred_transform_ov)
                    all_re_ov.append(re_ov.float().item())
                    all_te_ov.append(te_ov.float().item())
                    all_rr_ov.append(rr_ov.float().item())
            
            if self.num_gen_samples is not None and i == self.num_gen_samples - 1:
                break
        
        metrics = {
            "dist_error": torch.tensor(dist_error).mean().detach().item(),
            "RRE": torch.tensor(all_re).mean().item(),
            "RTE": torch.tensor(all_te).mean().item(),
            "RR": torch.tensor(all_rr).mean().item(),
        }
        if len(dist_ov_error) > 0:
            metrics.update({
                "dist_ov_error": torch.tensor(dist_ov_error).mean().detach().item(),
                "RRE_ov": torch.tensor(all_re_ov).mean().item(),
                "RTE_ov": torch.tensor(all_te_ov).mean().item(),
                "RR_ov": torch.tensor(all_rr_ov).mean().item(),
            })
        if len(all_re_w) > 0:
            metrics.update({
                "RRE_w": torch.tensor(all_re_w).mean().item(),
                "RTE_w": torch.tensor(all_te_w).mean().item(),
                "RR_w": torch.tensor(all_rr_w).mean().item(),
            })
        if len(overlap_ratios) > 0:
            metrics["overlap_ratio"] = torch.tensor(overlap_ratios).mean().item()
        
        return metrics
    
    def compute_transform_error(self, gt_transform, pred_transform):
        return self.evaluator(gt_transform, pred_transform)
