from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _as_range(value, default):
    if value is None:
        value = default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 0:
            return default
        if len(value) == 1:
            value = value[0]
        else:
            low, high = float(value[0]), float(value[1])
            return min(low, high), max(low, high)
    value = float(value)
    return value, value


def _sample_uniform_range(value, default):
    low, high = _as_range(value, default)
    if low == high:
        return low
    return np.random.uniform(low, high)


def _normalize_probs(items):
    probs = np.array([max(float(item[-1]), 0.0) for item in items], dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(items), dtype=np.float64) / len(items)
    else:
        probs = probs / total
    return probs


def _rotation_from_degrees(roll, pitch, yaw):
    angles = [
        np.random.uniform(-float(roll), float(roll)),
        np.random.uniform(-float(pitch), float(pitch)),
        np.random.uniform(-float(yaw), float(yaw)),
    ]
    return R.from_euler("xyz", angles, degrees=True).as_matrix().astype(np.float32)


def _unit_random_vector():
    normal = np.random.normal(size=3).astype(np.float32)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return normal / norm


def _keep_count(num_points, keep_ratio, min_points):
    if num_points <= 0:
        return 0
    keep = int(round(num_points * float(keep_ratio)))
    keep = max(int(min_points), keep)
    return min(num_points, keep)


def random_sphere_crop(points, keep_ratio, min_points):
    num_points = points.shape[0]
    keep = _keep_count(num_points, keep_ratio, min_points)
    if keep >= num_points:
        return points

    center = points[np.random.randint(num_points)]
    distances = np.sum((points - center) ** 2, axis=1)
    indices = np.argpartition(distances, keep - 1)[:keep]
    return points[indices]


def random_halfspace_crop(points, keep_ratio, min_points):
    num_points = points.shape[0]
    keep = _keep_count(num_points, keep_ratio, min_points)
    if keep >= num_points:
        return points

    normal = _unit_random_vector()
    projections = points @ normal
    if np.random.rand() < 0.5:
        indices = np.argpartition(projections, keep - 1)[:keep]
    else:
        indices = np.argpartition(-projections, keep - 1)[:keep]
    return points[indices]


def random_block_dropout(points, keep_ratio, min_points):
    num_points = points.shape[0]
    keep = _keep_count(num_points, keep_ratio, min_points)
    if keep >= num_points:
        return points

    drop = num_points - keep
    center = points[np.random.randint(num_points)]
    distances = np.sum((points - center) ** 2, axis=1)
    drop_indices = np.argpartition(distances, drop - 1)[:drop]
    keep_mask = np.ones(num_points, dtype=bool)
    keep_mask[drop_indices] = False
    return points[keep_mask]


class ThreeDMatchAugmentor:
    """Configurable train-time augmentation for 3DMatch point cloud pairs.

    Points use row-vector coordinates throughout the data pipeline:
    transformed_src = src @ Tr[:3, :3].T + Tr[:3, 3].
    """

    def __init__(self, augment_prob=1.0, config=None):
        self.augment_prob = float(augment_prob or 0.0)
        self.config = config
        self.enabled = bool(_cfg_get(config, "enabled", True)) and self.augment_prob > 0.0

        self.pose_cfg = _cfg_get(config, "pose", None)
        self.pose_enabled = bool(_cfg_get(self.pose_cfg, "enabled", False))
        self.source_prob = float(_cfg_get(self.pose_cfg, "source_prob", 0.0))
        self.global_prob = float(_cfg_get(self.pose_cfg, "global_prob", 0.0))
        self.pose_modes = self._load_pose_modes(_cfg_get(self.pose_cfg, "modes", None))
        self.global_cfg = _cfg_get(self.pose_cfg, "global", None)

        self.crop_cfg = _cfg_get(config, "crop", None)
        self.crop_enabled = bool(_cfg_get(self.crop_cfg, "enabled", False))
        self.crop_prob = float(_cfg_get(self.crop_cfg, "prob", 0.0))
        self.crop_min_points = int(_cfg_get(self.crop_cfg, "min_points", 2048))
        self.crop_modes = self._load_crop_modes(_cfg_get(self.crop_cfg, "modes", None))

    def __call__(self, ref_points, src_points, transform):
        if not self.enabled or np.random.rand() >= self.augment_prob:
            return ref_points, src_points, transform

        ref_points = ref_points.astype(np.float32, copy=False)
        src_points = src_points.astype(np.float32, copy=False)
        transform = transform.astype(np.float32, copy=True)

        if self.pose_enabled:
            if self.source_prob > 0.0 and np.random.rand() < self.source_prob:
                rotation = self.sample_source_rotation()
                src_points, transform = apply_source_rotation(src_points, transform, rotation)

            if self.global_prob > 0.0 and np.random.rand() < self.global_prob:
                rotation = self.sample_global_rotation()
                ref_points, src_points, transform = apply_global_rotation(
                    ref_points, src_points, transform, rotation
                )

        if self.crop_enabled and self.crop_prob > 0.0 and np.random.rand() < self.crop_prob:
            ref_points = self.apply_random_crop(ref_points)
            src_points = self.apply_random_crop(src_points)

        return (
            ref_points.astype(np.float32, copy=False),
            src_points.astype(np.float32, copy=False),
            transform.astype(np.float32, copy=False),
        )

    def _load_pose_modes(self, modes_cfg):
        if not modes_cfg:
            modes_cfg = [
                {"name": "legacy", "prob": 1.0, "roll": 45.0, "pitch": 60.0, "yaw": 0.0}
            ]
        modes = []
        for i, mode_cfg in enumerate(modes_cfg):
            modes.append(
                (
                    str(_cfg_get(mode_cfg, "name", f"mode_{i}")),
                    mode_cfg,
                    float(_cfg_get(mode_cfg, "prob", 1.0)),
                )
            )
        return modes

    def _load_crop_modes(self, modes_cfg):
        if not modes_cfg:
            modes_cfg = {
                "sphere": {"prob": 1.0, "keep_ratio": [0.5, 0.9]},
            }

        if isinstance(modes_cfg, Mapping):
            iterable = modes_cfg.items()
        else:
            try:
                iterable = modes_cfg.items()
            except AttributeError:
                iterable = []
                for i, mode_cfg in enumerate(modes_cfg):
                    name = _cfg_get(mode_cfg, "name", f"mode_{i}")
                    iterable.append((name, mode_cfg))

        modes = []
        for name, mode_cfg in iterable:
            modes.append((str(name), mode_cfg, float(_cfg_get(mode_cfg, "prob", 1.0))))
        if len(modes) == 0:
            modes.append(("sphere", {"keep_ratio": [0.5, 0.9]}, 1.0))
        return modes

    def sample_source_rotation(self):
        probs = _normalize_probs(self.pose_modes)
        mode_index = np.random.choice(len(self.pose_modes), p=probs)
        _, mode_cfg, _ = self.pose_modes[mode_index]
        return _rotation_from_degrees(
            _cfg_get(mode_cfg, "roll", 0.0),
            _cfg_get(mode_cfg, "pitch", 0.0),
            _cfg_get(mode_cfg, "yaw", 0.0),
        )

    def sample_global_rotation(self):
        return _rotation_from_degrees(
            _cfg_get(self.global_cfg, "roll", 0.0),
            _cfg_get(self.global_cfg, "pitch", 0.0),
            _cfg_get(self.global_cfg, "yaw", 0.0),
        )

    def apply_random_crop(self, points):
        if points.shape[0] <= self.crop_min_points:
            return points

        probs = _normalize_probs(self.crop_modes)
        mode_index = np.random.choice(len(self.crop_modes), p=probs)
        mode_name, mode_cfg, _ = self.crop_modes[mode_index]
        keep_ratio = _sample_uniform_range(_cfg_get(mode_cfg, "keep_ratio", None), [0.5, 0.9])

        if mode_name == "sphere":
            cropped = random_sphere_crop(points, keep_ratio, self.crop_min_points)
        elif mode_name == "halfspace":
            cropped = random_halfspace_crop(points, keep_ratio, self.crop_min_points)
        elif mode_name == "block_dropout":
            cropped = random_block_dropout(points, keep_ratio, self.crop_min_points)
        else:
            cropped = random_sphere_crop(points, keep_ratio, self.crop_min_points)

        if cropped.shape[0] < self.crop_min_points:
            return points
        return cropped


def apply_source_rotation(src_points, transform, rotation):
    src_points = src_points @ rotation
    transform = transform.copy()
    transform[:3, :3] = transform[:3, :3] @ rotation
    return src_points, transform


def apply_global_rotation(ref_points, src_points, transform, rotation):
    ref_points = ref_points @ rotation
    src_points = src_points @ rotation

    transform = transform.copy()
    base_rotation = transform[:3, :3].copy()
    base_translation = transform[:3, 3].copy()
    transform[:3, :3] = rotation.T @ base_rotation @ rotation
    transform[:3, 3] = base_translation @ rotation
    return ref_points, src_points, transform
