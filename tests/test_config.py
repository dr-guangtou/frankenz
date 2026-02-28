"""Tests for frankenz.config — dataclass hierarchy + YAML serde."""

import math

import pytest
import yaml

from frankenz.config import (
    FrankenzConfig, ModelConfig, TransformConfig, PriorConfig,
    ZGridConfig, PDFConfig, DataConfig, KDTreeConfig,
)


class TestDefaults:
    """Default config values are sane."""

    def test_top_level_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.verbose is True
        assert cfg.seed is None

    def test_model_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.model.backend == "knn"
        assert cfg.model.k_tree == 25
        assert cfg.model.k_point == 20
        assert cfg.model.free_scale is False

    def test_transform_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.transform.type == "luptitude"
        assert cfg.transform.zeropoints == 1.0
        assert cfg.transform.skynoise == [1.0]

    def test_zgrid_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.zgrid.z_start == 0.0
        assert cfg.zgrid.z_end == 7.0
        assert cfg.zgrid.z_delta == 0.01

    def test_kdtree_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.model.kdtree.leafsize == 50
        assert math.isinf(cfg.model.kdtree.distance_upper_bound)

    def test_prior_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.prior.type == "uniform"

    def test_data_defaults(self):
        cfg = FrankenzConfig()
        assert cfg.data.format == "csv"
        assert cfg.data.flux_columns == []
        assert cfg.data.redshift_column == "z"


class TestDictRoundtrip:
    """to_dict / from_dict roundtrip."""

    def test_roundtrip_defaults(self):
        cfg = FrankenzConfig()
        d = cfg.to_dict()
        cfg2 = FrankenzConfig.from_dict(d)
        assert cfg2.to_dict() == d

    def test_roundtrip_custom_values(self):
        cfg = FrankenzConfig(
            model=ModelConfig(backend="bruteforce", k_tree=10),
            transform=TransformConfig(type="magnitude", zeropoints=3631.0),
            seed=42,
        )
        d = cfg.to_dict()
        cfg2 = FrankenzConfig.from_dict(d)
        assert cfg2.model.backend == "bruteforce"
        assert cfg2.model.k_tree == 10
        assert cfg2.transform.type == "magnitude"
        assert cfg2.seed == 42

    def test_from_dict_ignores_unknown_keys(self):
        d = {"model": {"backend": "knn", "bogus_key": 999}, "verbose": True}
        cfg = FrankenzConfig.from_dict(d)
        assert cfg.model.backend == "knn"

    def test_from_dict_none_returns_defaults(self):
        cfg = FrankenzConfig.from_dict(None)
        assert cfg.model.backend == "knn"

    def test_from_dict_partial(self):
        d = {"model": {"k_tree": 50}}
        cfg = FrankenzConfig.from_dict(d)
        assert cfg.model.k_tree == 50
        assert cfg.model.backend == "knn"  # other defaults preserved


class TestYAMLRoundtrip:
    """YAML file load/save roundtrip."""

    def test_yaml_roundtrip(self, tmp_path):
        cfg = FrankenzConfig(
            model=ModelConfig(backend="bruteforce", free_scale=True),
            transform=TransformConfig(
                type="luptitude",
                skynoise=[0.1, 0.2, 0.3, 0.4, 0.5],
            ),
            seed=123,
        )
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)
        cfg2 = FrankenzConfig.from_yaml(path)
        assert cfg2.model.backend == "bruteforce"
        assert cfg2.model.free_scale is True
        assert cfg2.transform.skynoise == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert cfg2.seed == 123

    def test_yaml_is_human_readable(self, tmp_path):
        cfg = FrankenzConfig()
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)
        text = path.read_text()
        assert "model:" in text
        assert "transform:" in text
        # Not a single-line dump
        assert text.count("\n") > 10

    def test_yaml_creates_parent_dirs(self, tmp_path):
        cfg = FrankenzConfig()
        path = tmp_path / "subdir" / "deep" / "config.yaml"
        cfg.to_yaml(path)
        assert path.exists()


class TestOverride:
    """Config override mechanism."""

    def test_override_flat(self):
        cfg = FrankenzConfig()
        cfg2 = cfg.override({"verbose": False, "seed": 42})
        assert cfg2.verbose is False
        assert cfg2.seed == 42
        # Original unchanged
        assert cfg.verbose is True

    def test_override_nested(self):
        cfg = FrankenzConfig()
        cfg2 = cfg.override({"model": {"backend": "bruteforce", "k_tree": 10}})
        assert cfg2.model.backend == "bruteforce"
        assert cfg2.model.k_tree == 10
        # Other model defaults preserved
        assert cfg2.model.k_point == 20

    def test_override_deep_nested(self):
        cfg = FrankenzConfig()
        cfg2 = cfg.override({"model": {"kdtree": {"leafsize": 100}}})
        assert cfg2.model.kdtree.leafsize == 100
        assert cfg2.model.backend == "knn"  # preserved


class TestInfHandling:
    """YAML roundtrip with infinity values."""

    def test_inf_survives_yaml(self, tmp_path):
        cfg = FrankenzConfig()
        assert math.isinf(cfg.model.kdtree.distance_upper_bound)
        path = tmp_path / "config.yaml"
        cfg.to_yaml(path)
        cfg2 = FrankenzConfig.from_yaml(path)
        assert math.isinf(cfg2.model.kdtree.distance_upper_bound)
