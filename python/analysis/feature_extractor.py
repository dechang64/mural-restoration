"""
Mural style feature extraction using DINOv2.

Extracts 768-dimensional feature vectors from mural images using
Meta's DINOv2 self-supervised vision transformer. These features
capture artistic style, color palette, and compositional patterns
for similarity search across mural collections.

Supports two modes:
    - 'dinov2': Full DINOv2 features (768-dim, requires transformers)
    - 'legacy': Fallback ResNet-18 features (512-dim, requires torchvision)
    - 'mock':   Deterministic mock features for testing (configurable dim)
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MuralFeature:
    """Feature vector for a mural image."""
    mural_id: str
    cave: str = ""
    wall: str = ""
    dynasty: str = ""
    feature: np.ndarray = field(default_factory=lambda: np.zeros(768, dtype=np.float32))
    dim: int = 768

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mural_id": self.mural_id,
            "cave": self.cave,
            "wall": self.wall,
            "dynasty": self.dynasty,
            "dim": self.dim,
        }


class MuralFeatureExtractor:
    """Extract style features from mural images using DINOv2.

    Modes:
        dinov2  - DINOv2 ViT-B/14, 768-dim (default)
        legacy  - ResNet-18 avgpool, 512-dim
        mock    - Deterministic hash-based features (for testing)
    """

    DIMENSIONS = {
        "dinov2": 768,
        "legacy": 512,
        "mock": 768,
    }

    def __init__(self, mode: str = "mock", dim: Optional[int] = None):
        if mode not in self.DIMENSIONS:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {list(self.DIMENSIONS.keys())}")
        self.mode = mode
        self.dim = dim or self.DIMENSIONS[mode]
        self._model = None
        self._preprocess = None

        if mode == "dinov2":
            self._load_dinov2()
        elif mode == "legacy":
            self._load_resnet()

    def _load_dinov2(self):
        """Load DINOv2 ViT-B/14 model."""
        try:
            import torch
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained("facebook/dinov2-base")
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
        except ImportError:
            raise ImportError(
                "DINOv2 mode requires 'transformers' and 'torch'. "
                "Install with: pip install transformers torch"
            )

    def _load_resnet(self):
        """Load ResNet-18 for legacy feature extraction."""
        try:
            import torch
            import torchvision.models as models
            resnet = models.resnet18(pretrained=False)
            self._model = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
        except ImportError:
            raise ImportError(
                "Legacy mode requires 'torch' and 'torchvision'. "
                "Install with: pip install torch torchvision"
            )

    def extract(self, image: np.ndarray, mural_id: str = "",
                cave: str = "", wall: str = "", dynasty: str = "") -> MuralFeature:
        """Extract features from a mural image.

        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255
            mural_id: Unique identifier for the mural
            cave: Cave number (e.g., "45")
            wall: Wall orientation (e.g., "north", "south")
            dynasty: Dynasty/period (e.g., "Tang", "Song")

        Returns:
            MuralFeature with 768-dim (DINOv2) or 512-dim (ResNet) feature vector
        """
        if self.mode == "mock":
            return self._extract_mock(image, mural_id, cave, wall, dynasty)
        elif self.mode == "dinov2":
            return self._extract_dinov2(image, mural_id, cave, wall, dynasty)
        else:
            return self._extract_resnet(image, mural_id, cave, wall, dynasty)

    def _extract_mock(self, image: np.ndarray, mural_id: str,
                      cave: str, wall: str, dynasty: str) -> MuralFeature:
        """Deterministic mock features based on image hash."""
        rng = np.random.RandomState(abs(hash(mural_id or str(image.tobytes()))) % (2**31))
        feature = rng.randn(self.dim).astype(np.float32)
        feature = feature / (np.linalg.norm(feature) + 1e-8)
        return MuralFeature(
            mural_id=mural_id, cave=cave, wall=wall, dynasty=dynasty,
            feature=feature, dim=self.dim,
        )

    def _extract_dinov2(self, image: np.ndarray, mural_id: str,
                        cave: str, wall: str, dynasty: str) -> MuralFeature:
        """Extract features using DINOv2."""
        import torch
        from PIL import Image
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.fromarray(image.astype(np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            outputs = self._model(img_tensor)
            feature = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        feature = feature.astype(np.float32)
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return MuralFeature(
            mural_id=mural_id, cave=cave, wall=wall, dynasty=dynasty,
            feature=feature, dim=len(feature),
        )

    def _extract_resnet(self, image: np.ndarray, mural_id: str,
                        cave: str, wall: str, dynasty: str) -> MuralFeature:
        """Extract features using ResNet-18."""
        import torch
        from PIL import Image
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        pil_img = Image.fromarray(image.astype(np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            feature = self._model(img_tensor).squeeze().cpu().numpy()

        feature = feature.astype(np.float32).flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return MuralFeature(
            mural_id=mural_id, cave=cave, wall=wall, dynasty=dynasty,
            feature=feature, dim=len(feature),
        )

    def extract_batch(self, images: List[np.ndarray],
                      mural_ids: Optional[List[str]] = None,
                      **kwargs) -> List[MuralFeature]:
        """Extract features from a batch of mural images."""
        if mural_ids is None:
            mural_ids = [f"mural_{i}" for i in range(len(images))]
        return [
            self.extract(img, mural_id=mid, **kwargs)
            for img, mid in zip(images, mural_ids)
        ]
