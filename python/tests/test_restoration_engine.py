"""Tests for MuralRestorationEngine."""

import numpy as np
import pytest
from analysis.restoration_engine import (
    MuralRestorationEngine, RestorationResult, RestorationMethod,
)


class TestRestorationResult:
    def test_defaults(self):
        result = RestorationResult(mural_id="test", method="mock")
        assert result.mural_id == "test"
        assert result.confidence == 0.0
        assert result.defects_restored == 0
        assert result.image_size == (100, 100)

    def test_image_size(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        result = RestorationResult(mural_id="test", method="mock", restored_image=img)
        assert result.image_size == (200, 300)


class TestMuralRestorationEngine:
    def test_mock_mode(self):
        engine = MuralRestorationEngine(mode="mock")
        assert engine.mode == "mock"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            MuralRestorationEngine(mode="invalid")

    def test_restore_with_mask(self):
        engine = MuralRestorationEngine(mode="mock")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:50] = 255
        result = engine.restore(image, mask, mural_id="cave_45")
        assert isinstance(result, RestorationResult)
        assert result.mural_id == "cave_45"
        assert result.restored_image.shape == (100, 100, 3)
        assert result.defect_mask.shape == (100, 100)
        assert result.confidence > 0
        assert result.processing_time_ms > 0

    def test_restore_with_reference(self):
        engine = MuralRestorationEngine(mode="mock")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        result = engine.restore(image, mask, mural_id="cave_45",
                                reference_id="cave_45_ref")
        assert result.reference_id == "cave_45_ref"

    def test_restore_empty_mask(self):
        engine = MuralRestorationEngine(mode="mock")
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = engine.restore(image, mask, mural_id="test")
        # No defects to restore
        assert result.defects_restored == 0

    def test_restore_from_detection(self):
        from analysis.defect_detector import (
            MuralDefectDetector, DefectBox,
        )
        engine = MuralRestorationEngine(mode="mock")
        det = MuralDefectDetector(mode="mock")
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        detection = det.detect(image, mural_id="cave_45")
        result = engine.restore_from_detection(image, detection, mural_id="cave_45")
        assert isinstance(result, RestorationResult)
        assert result.mural_id == "cave_45"

    def test_restoration_methods(self):
        assert len(RestorationMethod) == 4
        assert RestorationMethod.INPAINTING.value == "inpainting"
        assert RestorationMethod.TEXTURE_SYNTHESIS.value == "texture_synthesis"
