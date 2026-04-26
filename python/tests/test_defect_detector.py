"""Tests for MuralDefectDetector."""

import numpy as np
import pytest
from analysis.defect_detector import (
    MuralDefectDetector, DefectBox, DetectionResult,
    DefectType, DEFECT_NAMES_CN, DEFECT_NAMES_EN,
)


class TestDefectBox:
    def test_properties(self):
        box = DefectBox(x1=10, y1=20, x2=50, y2=60, confidence=0.9, class_id=0)
        assert box.class_name == "flaking"
        assert box.class_name_cn == "起甲"
        assert box.area == 40 * 40
        assert box.center == (30.0, 40.0)

    def test_custom_class_name(self):
        box = DefectBox(x1=0, y1=0, x2=10, y2=10, confidence=0.5, class_id=3,
                        class_name="crack", class_name_cn="裂隙")
        assert box.class_name == "crack"
        assert box.class_name_cn == "裂隙"

    def test_out_of_range_class(self):
        box = DefectBox(x1=0, y1=0, x2=10, y2=10, confidence=0.5, class_id=99)
        assert box.class_name == "unknown"
        assert box.class_name_cn == "未知"


class TestDetectionResult:
    def test_empty(self):
        result = DetectionResult(mural_id="test", image_size=(100, 100))
        assert result.num_defects == 0
        assert result.defect_summary == {}
        assert result.health_score == 100.0
        assert not result.has_critical

    def test_with_defects(self):
        defects = [
            DefectBox(0, 0, 10, 10, 0.9, 0),
            DefectBox(20, 20, 30, 30, 0.8, 0),
            DefectBox(50, 50, 60, 60, 0.7, 1),  # saline = critical
        ]
        result = DetectionResult(mural_id="test", image_size=(100, 100), defects=defects)
        assert result.num_defects == 3
        assert result.defect_summary == {"起甲": 2, "酥碱": 1}
        assert result.has_critical
        assert result.health_score < 100.0

    def test_health_score(self):
        # 10% damage area
        defects = [DefectBox(0, 0, 100, 10, 0.9, 0)]
        result = DetectionResult(mural_id="test", image_size=(100, 100), defects=defects)
        assert result.health_score == 90.0


class TestMuralDefectDetector:
    def test_mock_mode(self):
        det = MuralDefectDetector(mode="mock")
        assert det.mode == "mock"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            MuralDefectDetector(mode="invalid")

    def test_detect(self):
        det = MuralDefectDetector(mode="mock")
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = det.detect(image, mural_id="cave_45")
        assert isinstance(result, DetectionResult)
        assert result.mural_id == "cave_45"
        assert result.image_size == (512, 512)
        assert isinstance(result.defects, list)

    def test_detect_deterministic(self):
        det = MuralDefectDetector(mode="mock")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        r1 = det.detect(image, mural_id="same")
        r2 = det.detect(image, mural_id="same")
        assert r1.num_defects == r2.num_defects
        for d1, d2 in zip(r1.defects, r2.defects):
            assert d1.class_id == d2.class_id

    def test_detect_batch(self):
        det = MuralDefectDetector(mode="mock")
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = det.detect_batch(images, mural_ids=["a", "b", "c"])
        assert len(results) == 3
        assert results[0].mural_id == "a"

    def test_defect_types(self):
        assert len(DefectType) == 6
        assert DefectType.FLAKING.label_cn == "起甲"
        assert DefectType.SALINE.severity == "critical"
        assert DefectType.FADING.severity == "minor"

    def test_defect_names(self):
        assert len(DEFECT_NAMES_CN) == 6
        assert len(DEFECT_NAMES_EN) == 6
        assert DEFECT_NAMES_CN[0] == "起甲"
        assert DEFECT_NAMES_EN[3] == "cracking"
