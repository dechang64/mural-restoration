"""
Mural defect detection using YOLOv11.

Detects 6 types of mural defects:
    0: flaking    (起甲) — paint layer detachment
    1: saline     (酥碱) — salt crystallization damage
    2: hollowing  (空鼓) — delamination from wall
    3: cracking   (裂隙) — structural cracks
    4: fading     (褪色) — pigment loss/fading
    5: mold       (霉变) — biological contamination

Supports two modes:
    - 'yolo':   YOLOv11 detection (requires ultralytics)
    - 'mock':   Deterministic mock detection for testing
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import IntEnum


class DefectType(IntEnum):
    FLAKING = 0     # 起甲
    SALINE = 1      # 酥碱
    HOLLOWING = 2   # 空鼓
    CRACKING = 3    # 裂隙
    FADING = 4      # 褪色
    MOLD = 5        # 霉变

    @property
    def label_cn(self) -> str:
        return ["起甲", "酥碱", "空鼓", "裂隙", "褪色", "霉变"][self]

    @property
    def label_en(self) -> str:
        return ["flaking", "saline", "hollowing", "cracking", "fading", "mold"][self]

    @property
    def severity(self) -> str:
        return ["major", "critical", "critical", "major", "minor", "major"][self]


DEFECT_NAMES_CN = ["起甲", "酥碱", "空鼓", "裂隙", "褪色", "霉变"]
DEFECT_NAMES_EN = ["flaking", "saline", "hollowing", "cracking", "fading", "mold"]


@dataclass
class DefectBox:
    """A single detected defect bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str = ""
    class_name_cn: str = ""

    def __post_init__(self):
        if not self.class_name:
            self.class_name = DEFECT_NAMES_EN[self.class_id] if 0 <= self.class_id < 6 else "unknown"
        if not self.class_name_cn:
            self.class_name_cn = DEFECT_NAMES_CN[self.class_id] if 0 <= self.class_id < 6 else "未知"

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """Detection result for a single mural image."""
    mural_id: str
    image_size: Tuple[int, int] = (0, 0)
    defects: List[DefectBox] = field(default_factory=list)
    num_defects: int = 0
    defect_summary: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.num_defects = len(self.defects)
        self.defect_summary = {}
        for d in self.defects:
            name = d.class_name_cn
            self.defect_summary[name] = self.defect_summary.get(name, 0) + 1

    @property
    def has_critical(self) -> bool:
        return any(d.class_id in (1, 2) for d in self.defects)  # saline, hollowing

    @property
    def health_score(self) -> float:
        """0-100 health score (100 = no defects)."""
        if not self.defects:
            return 100.0
        total_area = self.image_size[0] * self.image_size[1]
        if total_area == 0:
            return 50.0
        damaged = sum(d.area for d in self.defects)
        ratio = min(damaged / total_area, 1.0)
        return round(100.0 * (1.0 - ratio), 1)


class MuralDefectDetector:
    """Detect defects in mural images using YOLOv11.

    Modes:
        yolo  - YOLOv11 nano model (requires ultralytics)
        mock  - Deterministic mock detection for testing
    """

    def __init__(self, mode: str = "mock", conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        if mode not in ("yolo", "mock"):
            raise ValueError("mode must be 'yolo' or 'mock'")
        self.mode = mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None

        if mode == "yolo":
            self._load_yolo()

    def _load_yolo(self):
        """Load YOLOv11 nano model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolo11n.pt")
        except ImportError:
            raise ImportError(
                "YOLO mode requires 'ultralytics'. "
                "Install with: pip install ultralytics"
            )

    def detect(self, image: np.ndarray, mural_id: str = "") -> DetectionResult:
        """Detect defects in a mural image.

        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255
            mural_id: Unique identifier for the mural

        Returns:
            DetectionResult with bounding boxes and summary
        """
        if self.mode == "mock":
            return self._detect_mock(image, mural_id)
        else:
            return self._detect_yolo(image, mural_id)

    def _detect_mock(self, image: np.ndarray, mural_id: str) -> DetectionResult:
        """Deterministic mock detection based on image hash."""
        h, w = image.shape[:2]
        rng = np.random.RandomState(abs(hash(mural_id or "default")) % (2**31))
        n_defects = rng.randint(0, 4)
        defects = []
        for _ in range(n_defects):
            class_id = rng.randint(0, 6)
            cx, cy = rng.uniform(0.1, 0.9, 2)
            bw, bh = rng.uniform(0.05, 0.2, 2)
            defects.append(DefectBox(
                x1=float(max(0, cx - bw/2) * w),
                y1=float(max(0, cy - bh/2) * h),
                x2=float(min(w, (cx + bw/2) * w)),
                y2=float(min(h, (cy + bh/2) * h)),
                confidence=float(rng.uniform(0.3, 0.95)),
                class_id=int(class_id),
            ))
        return DetectionResult(
            mural_id=mural_id,
            image_size=(h, w),
            defects=defects,
        )

    def _detect_yolo(self, image: np.ndarray, mural_id: str) -> DetectionResult:
        """Detect defects using YOLOv11."""
        results = self._model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        h, w = image.shape[:2]
        defects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                defects.append(DefectBox(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                ))
        return DetectionResult(
            mural_id=mural_id,
            image_size=(h, w),
            defects=defects,
        )

    def detect_batch(self, images: List[np.ndarray],
                     mural_ids: Optional[List[str]] = None) -> List[DetectionResult]:
        """Detect defects in a batch of mural images."""
        if mural_ids is None:
            mural_ids = [f"mural_{i}" for i in range(len(images))]
        return [
            self.detect(img, mural_id=mid)
            for img, mid in zip(images, mural_ids)
        ]
