import torch
from tqdm import tqdm
from src.models.pipeline import PointGenPipeline
from src.utils.point_cloud_utils import weighted_svd

from src.models.pipeline import Evaluator

class DiffusionEvaluator:    
    def __init__(self, cfg, processor=None, noise_scheduler=None):
        self.evaluator = Evaluator(**cfg.eval)
        self.num_gen_samples = cfg.num_gen_samples
        self.num_inference_steps = cfg.num_inference_steps
        self.inference_type = cfg.inference_type
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
        
        progress_bar = tqdm(dataloader, desc="Evaluation")
        
        for i, data_dict in enumerate(progress_bar):
            pred_points, tgt_points, ref_points, tgt_points_corr, src_points, gt_overlap = pipeline(
                data_dict, 
                num_inference_steps=self.num_inference_steps, 
                generator=generator
            )
            
            pred_points = pred_points.squeeze(0)
            pred_points = pred_points * torch.std(ref_points, dim=0) + torch.mean(ref_points, dim=0)
            
            per_point_dist_error = torch.norm(pred_points - tgt_points, dim=-1).mean()
            dist_error.append(per_point_dist_error)
            
            pred_transform = weighted_svd(src_points.squeeze(0), pred_points)
            te, re, rr = self.compute_transform_error(data_dict["Tr"].squeeze(0), pred_transform)
            all_te.append(te.float().item())
            all_re.append(re.float().item())
            all_rr.append(rr.float().item())

            # overlap_mask = gt_overlap > 0.5
            if False and overlap_mask.sum() > 0:
                ov_dist_error = torch.norm(pred_points[overlap_mask] - tgt_points[overlap_mask], dim=-1).mean()
                dist_ov_error.append(ov_dist_error)
                pred_transform_ov = weighted_svd(src_points.squeeze(0)[overlap_mask], pred_points[overlap_mask])
                te_ov, re_ov, rr_ov = self.compute_transform_error(data_dict["Tr"].squeeze(0), pred_transform_ov)
                all_re_ov.append(re_ov.float().item())
                all_te_ov.append(te_ov.float().item())
                all_rr_ov.append(rr_ov.float().item())

                ov_metrics = {
                    "dist_ov_error": torch.tensor(dist_ov_error).mean().detach().item(),
                    "RREO": torch.tensor(all_re_ov).mean().item(),
                    "RTEO": torch.tensor(all_te_ov).mean().item(),
                    "RRO": torch.tensor(all_rr_ov).mean().item()
                }
            
            if i == self.num_gen_samples - 1:
                break
        
        metrics = {
            "dist_error": torch.tensor(dist_error).mean().detach().item(),
            "RRE": torch.tensor(all_re).mean().item(),
            "RTE": torch.tensor(all_te).mean().item(),
            "RR": torch.tensor(all_rr).mean().item(),
        }
        
        return metrics
    
    def compute_transform_error(self, gt_transform, pred_transform):
        return self.evaluator(gt_transform, pred_transform)
