"""Tests for MuralFeatureExtractor."""

import numpy as np
import pytest
from analysis.feature_extractor import (
    MuralFeatureExtractor, MuralFeature,
)


class TestMuralFeatureExtractor:
    """Test suite for mural feature extraction."""

    def test_mock_mode_default_dim(self):
        ext = MuralFeatureExtractor(mode="mock")
        assert ext.mode == "mock"
        assert ext.dim == 768

    def test_mock_mode_custom_dim(self):
        ext = MuralFeatureExtractor(mode="mock", dim=128)
        assert ext.dim == 128

    def test_extract_single(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        feat = ext.extract(image, mural_id="cave_45_north")
        assert isinstance(feat, MuralFeature)
        assert feat.mural_id == "cave_45_north"
        assert feat.dim == 64
        assert feat.feature.shape == (64,)
        assert feat.feature.dtype == np.float32

    def test_extract_normalized(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat = ext.extract(image, mural_id="test")
        norm = np.linalg.norm(feat.feature)
        assert abs(norm - 1.0) < 1e-5

    def test_extract_deterministic(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat1 = ext.extract(image, mural_id="same_id")
        feat2 = ext.extract(image, mural_id="same_id")
        np.testing.assert_array_equal(feat1.feature, feat2.feature)

    def test_extract_different_ids(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat1 = ext.extract(image, mural_id="id_a")
        feat2 = ext.extract(image, mural_id="id_b")
        assert not np.array_equal(feat1.feature, feat2.feature)

    def test_extract_with_metadata(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat = ext.extract(image, mural_id="cave_45",
                           cave="45", wall="north", dynasty="Tang")
        assert feat.cave == "45"
        assert feat.wall == "north"
        assert feat.dynasty == "Tang"

    def test_extract_batch(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]
        feats = ext.extract_batch(images, mural_ids=["m1", "m2", "m3", "m4", "m5"])
        assert len(feats) == 5
        assert all(f.dim == 64 for f in feats)
        assert [f.mural_id for f in feats] == ["m1", "m2", "m3", "m4", "m5"]

    def test_extract_batch_auto_ids(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        images = [np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8) for _ in range(3)]
        feats = ext.extract_batch(images)
        assert len(feats) == 3
        assert feats[0].mural_id == "mural_0"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            MuralFeatureExtractor(mode="invalid")

    def test_to_dict(self):
        ext = MuralFeatureExtractor(mode="mock", dim=64)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        feat = ext.extract(image, mural_id="test", cave="45")
        d = feat.to_dict()
        assert d["mural_id"] == "test"
        assert d["cave"] == "45"
        assert d["dim"] == 64
