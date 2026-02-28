"""
Configuration system for frankenz using Python dataclasses + YAML.

Provides a hierarchical configuration with sensible defaults that can be
loaded from / saved to YAML files.
"""

import copy
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

import yaml

__all__ = [
    "TransformConfig", "KDTreeConfig", "ModelConfig", "PriorConfig",
    "ZGridConfig", "PDFConfig", "DataConfig", "FrankenzConfig",
]


@dataclass
class TransformConfig:
    """Transform configuration for photometric feature mapping."""
    type: str = "luptitude"
    zeropoints: float = 1.0
    skynoise: list = field(default_factory=lambda: [1.0])


@dataclass
class KDTreeConfig:
    """KDTree construction parameters."""
    leafsize: int = 50
    eps: float = 1e-3
    lp_norm: int = 2
    distance_upper_bound: float = float("inf")


@dataclass
class ModelConfig:
    """Model fitting parameters."""
    backend: str = "knn"
    k_tree: int = 25
    k_point: int = 20
    free_scale: bool = False
    ignore_model_err: bool = False
    dim_prior: bool = True
    track_scale: bool = False
    kdtree: KDTreeConfig = field(default_factory=KDTreeConfig)


@dataclass
class PriorConfig:
    """Prior probability configuration."""
    type: str = "uniform"
    k_tree: int = 25
    k_point: int = 20
    kdtree: KDTreeConfig = field(default_factory=KDTreeConfig)


@dataclass
class ZGridConfig:
    """Redshift grid specification."""
    z_start: float = 0.0
    z_end: float = 7.0
    z_delta: float = 0.01


@dataclass
class PDFConfig:
    """PDF construction parameters."""
    wt_thresh: float = 1e-3
    cdf_thresh: float = 2e-4


@dataclass
class DataConfig:
    """Data I/O configuration."""
    format: str = "csv"
    flux_columns: list = field(default_factory=list)
    flux_err_columns: list = field(default_factory=list)
    redshift_column: str = "z"
    redshift_err_column: str = "zerr"
    object_id_column: str = "object_id"


@dataclass
class FrankenzConfig:
    """Top-level frankenz configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    transform: TransformConfig = field(default_factory=TransformConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    zgrid: ZGridConfig = field(default_factory=ZGridConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    data: DataConfig = field(default_factory=DataConfig)
    verbose: bool = True
    seed: int = None

    def to_dict(self):
        """Convert config to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path):
        """Save config to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False,
                      sort_keys=False)

    @classmethod
    def from_dict(cls, d):
        """Create config from a plain dictionary, recursively constructing
        nested dataclasses."""
        return _dict_to_dataclass(cls, d)

    @classmethod
    def from_yaml(cls, path):
        """Load config from a YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def override(self, overrides):
        """Apply a flat or nested dict of overrides to this config.

        Returns a new FrankenzConfig with the overrides applied.
        """
        base = self.to_dict()
        _deep_update(base, overrides)
        return self.from_dict(base)


# --- Internal helpers ---

# Registry mapping dataclass field types to their classes for recursive
# from_dict construction.
_NESTED_TYPES = {
    "model": ModelConfig,
    "transform": TransformConfig,
    "prior": PriorConfig,
    "zgrid": ZGridConfig,
    "pdf": PDFConfig,
    "data": DataConfig,
    "kdtree": KDTreeConfig,
}


def _dict_to_dataclass(cls, d):
    """Recursively convert a dict into a dataclass instance."""
    if d is None:
        return cls()
    kwargs = {}
    valid_fields = {f.name for f in fields(cls)}
    for key, value in d.items():
        if key not in valid_fields:
            continue
        if key in _NESTED_TYPES and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(_NESTED_TYPES[key], value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_update(base, overrides):
    """Recursively merge overrides into base dict."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
