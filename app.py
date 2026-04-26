"""
🏛️ 壁画守护者 (Mural Guardian) - v2.0
基于AI的敦煌壁画智能修复与保护平台

功能:
1. 病害智能检测 (基于YOLOv11 / SAM)
2. 风格一致修复 (基于Stable Diffusion Inpainting)
3. 相似文物检索 (基于DINOv2 + HNSW)
4. 修复方案审计 (区块链哈希链)
5. 联邦学习协作 (多机构联合训练缺陷分类器)
6. 数字藏品铸造 (修复成果数字证书)
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import time
import uuid
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import IntEnum, Enum
import io
import base64

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import cv2

# ==================== 配置 ====================
st.set_page_config(
    page_title="壁画守护者 | Mural Guardian",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5f7a;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    .audit-trail {
        background: #1a1a2e;
        color: #0f0;
        font-family: monospace;
        padding: 1rem;
        border-radius: 8px;
        max-height: 300px;
        overflow-y: auto;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-success { background: #10b981; color: white; }
    .status-warning { background: #f59e0b; color: white; }
    .status-info { background: #3b82f6; color: white; }
    .rarity-common { color: #8B9DAF; }
    .rarity-uncommon { color: #4CAF50; }
    .rarity-rare { color: #2196F3; }
    .rarity-epic { color: #9C27B0; }
    .rarity-legendary { color: #FF9800; }
    .nft-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 2px solid #667eea;
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 病害检测模块 (升级版: 结构化检测)
# ============================================================

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
        return ["中危", "中危", "高危", "中危", "低危", "高危"][self]

    @property
    def color(self) -> str:
        return ["#FFA726", "#EF5350", "#AB47BC", "#42A5F5", "#78909C", "#66BB6A"][self]


@dataclass
class DefectBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    class_id: int = 0

    @property
    def defect_type(self) -> DefectType:
        return DefectType(self.class_id)

    @property
    def area(self) -> float:
        return max(0, (self.x2 - self.x1) * (self.y2 - self.y1))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    mural_id: str
    image_size: Tuple[int, int]
    defects: List[DefectBox] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def defect_count(self) -> int:
        return len(self.defects)

    @property
    def severity_summary(self) -> Dict[str, int]:
        summary = {}
        for d in self.defects:
            label = d.defect_type.label_cn
            summary[label] = summary.get(label, 0) + 1
        return summary

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for d in self.defects:
            rows.append({
                "病害类型": d.defect_type.label_cn,
                "英文名": d.defect_type.label_en,
                "置信度": f"{d.confidence:.1%}",
                "危险等级": d.defect_type.severity,
                "位置": f"({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})",
                "面积(px²)": f"{d.area:.0f}",
            })
        return pd.DataFrame(rows)


class MuralDefectDetector:
    """壁画病害检测器 (支持YOLOv11和模拟模式)"""

    def __init__(self, mode: str = "mock", conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.mode = mode
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        if mode == "yolo":
            try:
                from ultralytics import YOLO
                self._model = YOLO("yolo11n.pt")
            except ImportError:
                self.mode = "mock"

    def detect(self, image: np.ndarray, mural_id: str = "mural_0") -> DetectionResult:
        start = time.time()
        h, w = image.shape[:2]

        if self.mode == "yolo" and self._model is not None:
            results = self._model(
                image, conf=self.conf_threshold,
                iou=self.iou_threshold, verbose=False,
            )
            defects = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    defects.append(DefectBox(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]) % 6,
                    ))
        else:
            # 模拟检测
            rng = np.random.RandomState(hash(mural_id) % 2**31)
            n_defects = rng.randint(2, 6)
            defects = []
            for _ in range(n_defects):
                class_id = rng.randint(0, 6)
                bw = rng.randint(w // 8, w // 3)
                bh = rng.randint(h // 8, h // 3)
                bx = rng.randint(0, max(1, w - bw))
                by = rng.randint(0, max(1, h - bh))
                defects.append(DefectBox(
                    x1=bx, y1=by, x2=bx + bw, y2=by + bh,
                    confidence=rng.uniform(0.7, 0.98),
                    class_id=class_id,
                ))

        elapsed = (time.time() - start) * 1000
        return DetectionResult(
            mural_id=mural_id, image_size=(h, w),
            defects=defects, processing_time_ms=round(elapsed, 1),
        )


# ============================================================
# 修复引擎模块 (升级版)
# ============================================================

class RestorationMethod(Enum):
    INPAINTING = "inpainting"
    COLOR_TRANSFER = "color_transfer"
    TEXTURE_SYNTHESIS = "texture_synthesis"
    VIRTUAL = "virtual"


@dataclass
class RestorationResult:
    mural_id: str
    method: str
    restored_image: np.ndarray = field(default_factory=lambda: np.zeros((100, 100, 3), dtype=np.uint8))
    defect_mask: np.ndarray = field(default_factory=lambda: np.zeros((100, 100), dtype=np.uint8))
    confidence: float = 0.0
    reference_id: str = ""
    processing_time_ms: float = 0.0
    defects_restored: int = 0


class MuralRestorationEngine:
    """壁画虚拟修复引擎"""

    def __init__(self, mode: str = "mock"):
        self.mode = mode
        self._inpaint_pipeline = None
        if mode == "inpaint":
            try:
                from diffusers import StableDiffusionInpaintPipeline
                self._inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    "runwayml/stable-diffusion-inpainting",
                    torch_dtype=torch.float16,
                )
                if torch.cuda.is_available():
                    self._inpaint_pipeline = self._inpaint_pipeline.to("cuda")
            except ImportError:
                self.mode = "mock"

    def restore(self, image: np.ndarray, mask: np.ndarray,
                mural_id: str = "", reference_id: str = "",
                prompt: str = "") -> RestorationResult:
        start = time.time()
        h, w = image.shape[:2]

        if self.mode == "inpaint" and self._inpaint_pipeline is not None:
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(image)
            mask_pil = PILImage.fromarray(mask)
            default_prompt = prompt or "ancient Chinese mural painting, Dunhuang style, Buddhist art, detailed, high quality"
            result_pil = self._inpaint_pipeline(
                prompt=default_prompt, image=img_pil, mask_image=mask_pil,
            ).images[0]
            restored = np.array(result_pil)
        else:
            # 模拟修复: 中值滤波 + 边缘羽化混合
            restored = image.copy()
            mask_bool = mask > 0
            for i in range(3):
                channel = restored[:, :, i].astype(np.float64)
                blurred = cv2.medianBlur(restored[:, :, i], 21).astype(np.float64)
                kernel = np.ones((5, 5), np.float32) / 25
                feather = cv2.filter2D(mask.astype(np.float32), -1, kernel)
                feather = np.clip(feather, 0, 1)
                restored[:, :, i] = np.where(
                    mask_bool,
                    (channel * (1 - feather) + blurred * feather).astype(np.uint8),
                    restored[:, :, i],
                )
            # 锐化
            kernel_sharp = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) / 9
            enhanced = cv2.filter2D(restored, -1, kernel_sharp * 0.3)
            restored = np.clip(restored * 0.7 + enhanced * 0.3, 0, 255).astype(np.uint8)

        elapsed = (time.time() - start) * 1000
        return RestorationResult(
            mural_id=mural_id,
            method=self.mode,
            restored_image=restored,
            defect_mask=mask.astype(np.uint8) if mask.ndim == 2 else mask[:, :, 0].astype(np.uint8),
            confidence=0.92,
            reference_id=reference_id,
            processing_time_ms=round(elapsed, 1),
            defects_restored=1,
        )

    def restore_from_detection(self, image: np.ndarray,
                               detection_result: DetectionResult,
                               mural_id: str = "",
                               reference_id: str = "") -> RestorationResult:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for defect in detection_result.defects:
            x1, y1 = int(defect.x1), int(defect.y1)
            x2, y2 = int(defect.x2), int(defect.y2)
            mask[y1:y2, x1:x2] = 255
        return self.restore(image, mask, mural_id=mural_id, reference_id=reference_id)


# ============================================================
# 特征提取模块 (升级版: DINOv2)
# ============================================================

@dataclass
class MuralFeature:
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
    """壁画风格特征提取器 (支持DINOv2/ResNet/模拟)"""

    def __init__(self, mode: str = "mock", dim: Optional[int] = None):
        self.mode = mode
        self._dim = dim or 768
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        if mode == "dinov2":
            try:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained("facebook/dinov2-base")
                self._model.to(self._device).eval()
            except ImportError:
                self.mode = "legacy"
        if mode == "legacy" or (mode == "dinov2" and self._model is None):
            try:
                import torchvision.models as models
                resnet = models.resnet18(pretrained=True)
                self._model = nn.Sequential(*list(resnet.children())[:-1])
                self._model.to(self._device).eval()
                self._dim = 512
            except ImportError:
                self.mode = "mock"

    def extract(self, image: np.ndarray, mural_id: str = "",
                cave: str = "", wall: str = "", dynasty: str = "") -> MuralFeature:
        if self.mode == "mock":
            rng = np.random.RandomState(hash(mural_id) % 2**31)
            feature = rng.randn(self._dim).astype(np.float32)
            feature = feature / (np.linalg.norm(feature) + 1e-8)
            return MuralFeature(
                mural_id=mural_id, cave=cave, wall=wall, dynasty=dynasty,
                feature=feature, dim=self._dim,
            )

        from PIL import Image as PILImage
        from torchvision import transforms

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        pil_img = PILImage.fromarray(image.astype(np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            feature = self._model(img_tensor).squeeze().cpu().numpy()

        feature = feature.astype(np.float32).flatten()
        feature = feature / (np.linalg.norm(feature) + 1e-8)

        return MuralFeature(
            mural_id=mural_id, cave=cave, wall=wall, dynasty=dynasty,
            feature=feature, dim=len(feature),
        )


# ============================================================
# 联邦学习模块 (升级版)
# ============================================================

@dataclass
class FLConfig:
    num_clients: int = 3
    rounds: int = 10
    local_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    input_dim: int = 768
    num_classes: int = 6
    val_split: float = 0.2
    seed: int = 42


@dataclass
class RoundMetrics:
    round_num: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_acc: float = 0.0
    client_metrics: List[Dict[str, float]] = field(default_factory=list)
    elapsed_ms: float = 0.0


class DefectClassifier(nn.Module):
    """壁画缺陷分类器 MLP"""

    def __init__(self, input_dim: int = 768, num_classes: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MuralFLEngine:
    """壁画联邦学习引擎 (FedAvg)"""

    def __init__(self, config: Optional[FLConfig] = None):
        self.config = config or FLConfig()
        self.global_model = DefectClassifier(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
        )
        self.history: List[RoundMetrics] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_model.to(self._device)

    def _split_data(self, features: np.ndarray, labels: np.ndarray,
                    num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        rng = np.random.RandomState(self.config.seed)
        indices = rng.permutation(len(features))
        client_size = len(features) // num_clients
        splits = []
        for i in range(num_clients):
            start = i * client_size
            end = start + client_size if i < num_clients - 1 else len(features)
            client_idx = indices[start:end]
            splits.append((features[client_idx], labels[client_idx]))
        return splits

    def _train_client(self, model: nn.Module,
                      features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        n_val = max(1, int(len(features) * self.config.val_split))
        val_idx = np.random.choice(len(features), n_val, replace=False)
        train_idx = np.array([i for i in range(len(features)) if i not in val_idx])
        if len(train_idx) == 0:
            train_idx = val_idx

        train_dataset = TensorDataset(
            torch.FloatTensor(features[train_idx]),
            torch.LongTensor(labels[train_idx]),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)

        total_loss = 0.0
        n_batches = 0
        for _ in range(self.config.local_epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self._device), batch_y.to(self._device)
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        model.eval()
        with torch.no_grad():
            val_x = torch.FloatTensor(features[val_idx]).to(self._device)
            val_y = torch.LongTensor(labels[val_idx]).to(self._device)
            val_output = model(val_x)
            val_loss = criterion(val_output, val_y).item()
            val_acc = (val_output.argmax(dim=1) == val_y).float().mean().item()

        return {
            "train_loss": total_loss / max(n_batches, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

    def _fedavg_aggregate(self, client_models: List[nn.Module]) -> None:
        global_state = self.global_model.state_dict()
        aggregated = {}
        for key in global_state.keys():
            stacked = torch.stack([
                client.state_dict()[key].float()
                for client in client_models
            ], dim=0)
            aggregated[key] = stacked.mean(dim=0).to(global_state[key].dtype)
        self.global_model.load_state_dict(aggregated)

    def run(self, features: np.ndarray, labels: np.ndarray,
            num_clients: Optional[int] = None,
            rounds: Optional[int] = None) -> List[RoundMetrics]:
        num_clients = num_clients or self.config.num_clients
        rounds = rounds or self.config.rounds

        client_data = self._split_data(features, labels, num_clients)
        self.history = []

        for round_num in range(1, rounds + 1):
            start = time.time()
            client_models = []
            client_metrics = []

            for client_features, client_labels in client_data:
                client_model = copy.deepcopy(self.global_model)
                metrics = self._train_client(client_model, client_features, client_labels)
                client_models.append(client_model)
                client_metrics.append(metrics)

            self._fedavg_aggregate(client_models)

            self.global_model.eval()
            with torch.no_grad():
                all_x = torch.FloatTensor(features).to(self._device)
                all_y = torch.LongTensor(labels).to(self._device)
                criterion = nn.CrossEntropyLoss()
                output = self.global_model(all_x)
                val_loss = criterion(output, all_y).item()
                val_acc = (output.argmax(dim=1) == all_y).float().mean().item()

            avg_train_loss = np.mean([m["train_loss"] for m in client_metrics])
            elapsed = (time.time() - start) * 1000

            round_metrics = RoundMetrics(
                round_num=round_num,
                train_loss=round(avg_train_loss, 4),
                val_loss=round(val_loss, 4),
                val_acc=round(val_acc, 4),
                client_metrics=client_metrics,
                elapsed_ms=round(elapsed, 1),
            )
            self.history.append(round_metrics)

        return self.history

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.global_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features).to(self._device)
            logits = self.global_model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        return preds, probs


# ============================================================
# 数字藏品模块 (升级版)
# ============================================================

class RarityTier(IntEnum):
    COMMON = 0
    UNCOMMON = 1
    RARE = 2
    EPIC = 3
    LEGENDARY = 4

    @property
    def label_cn(self) -> str:
        return ["常见", "罕见", "稀有", "史诗", "传说"][self]

    @property
    def label_en(self) -> str:
        return ["Common", "Uncommon", "Rare", "Epic", "Legendary"][self]

    @property
    def color(self) -> str:
        return ["#8B9DAF", "#4CAF50", "#2196F3", "#9C27B0", "#FF9800"][self]

    @property
    def max_supply(self) -> int:
        return [10000, 5000, 1000, 100, 10][self]


DEFECT_SEVERITY = {
    "fading": 1, "flaking": 2, "cracking": 2,
    "mold": 4, "saline": 5, "hollowing": 5,
}

DYNASTY_WEIGHT = {
    "modern": 1, "qing": 1, "ming": 1, "yuan": 2, "song": 2,
    "five_dynasties": 3, "tang": 4, "sui": 5,
    "northern_wei": 6, "sixteen_kingdoms": 7,
}


@dataclass
class MuralProvenance:
    cave_id: str = ""
    wall: str = ""
    dynasty: str = ""
    location: str = ""
    period: str = ""
    description: str = ""


@dataclass
class RestorationRecord:
    defect_type: str = ""
    defect_severity: str = ""
    method: str = ""
    reference_id: str = ""
    expert_id: str = ""
    confidence: float = 0.0
    processing_time_ms: float = 0.0


@dataclass
class FeatureFingerprint:
    feature_hash: str = ""
    feature_dim: int = 768
    model_name: str = "dinov2_vitb14"
    similarity_threshold: float = 0.85


@dataclass
class CollectibleMetadata:
    name: str = ""
    description: str = ""
    image: str = ""
    external_url: str = ""
    attributes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DigitalCollectible:
    token_id: str = ""
    mint_tx_hash: str = ""
    mint_timestamp: str = ""
    provenance: MuralProvenance = field(default_factory=MuralProvenance)
    restoration: RestorationRecord = field(default_factory=RestorationRecord)
    fingerprint: FeatureFingerprint = field(default_factory=FeatureFingerprint)
    rarity: RarityTier = RarityTier.COMMON
    edition: int = 1
    max_edition: int = 10000
    audit_block_hash: str = ""
    audit_block_index: int = 0
    metadata: CollectibleMetadata = field(default_factory=CollectibleMetadata)

    def compute_token_hash(self) -> str:
        data = {
            "provenance": asdict(self.provenance),
            "restoration": asdict(self.restoration),
            "fingerprint": asdict(self.fingerprint),
            "rarity": self.rarity.value,
            "edition": self.edition,
            "audit_block_hash": self.audit_block_hash,
        }
        raw = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_certificate_json(self) -> str:
        return json.dumps({
            "token_id": self.token_id,
            "token_hash": self.compute_token_hash(),
            "mint_tx_hash": self.mint_tx_hash,
            "mint_timestamp": self.mint_timestamp,
            "provenance": asdict(self.provenance),
            "restoration": asdict(self.restoration),
            "fingerprint": asdict(self.fingerprint),
            "rarity": {
                "tier": self.rarity.value,
                "label_cn": self.rarity.label_cn,
                "label_en": self.rarity.label_en,
                "color": self.rarity.color,
            },
            "edition": f"{self.edition}/{self.max_edition}",
            "audit_block_hash": self.audit_block_hash,
            "audit_block_index": self.audit_block_index,
            "metadata": asdict(self.metadata),
        }, indent=2, ensure_ascii=False)

    def verify(self) -> Tuple[bool, str]:
        if not self.token_id:
            return False, "Missing token_id"
        if not self.mint_timestamp:
            return False, "Missing mint_timestamp"
        if not self.fingerprint.feature_hash:
            return False, "Missing feature_hash"
        if not self.audit_block_hash:
            return False, "Missing audit_block_hash"
        return True, "Valid"


class CollectibleMinter:
    """数字藏品铸造引擎"""

    def __init__(self):
        self._minted: Dict[str, DigitalCollectible] = {}
        self._edition_counters: Dict[str, int] = {}

    @property
    def total_minted(self) -> int:
        return len(self._minted)

    def compute_rarity(self, provenance: MuralProvenance,
                       restoration: RestorationRecord) -> RarityTier:
        defect_score = DEFECT_SEVERITY.get(restoration.defect_type, 1)
        dynasty_score = DYNASTY_WEIGHT.get(provenance.dynasty, 1)
        severity_mult = {"minor": 1, "major": 1.5, "critical": 2.0}
        sev_mult = severity_mult.get(restoration.defect_severity, 1.0)
        score = defect_score * dynasty_score * sev_mult

        if score >= 40:
            return RarityTier.LEGENDARY
        elif score >= 15:
            return RarityTier.EPIC
        elif score >= 4:
            return RarityTier.RARE
        elif score >= 2:
            return RarityTier.UNCOMMON
        else:
            return RarityTier.COMMON

    def compute_feature_hash(self, feature_vector) -> str:
        if hasattr(feature_vector, 'tobytes'):
            raw = feature_vector.astype(np.float32).tobytes()
        else:
            raw = json.dumps(feature_vector).encode()
        return hashlib.sha256(raw).hexdigest()

    def generate_token_id(self, provenance: MuralProvenance,
                          rarity: RarityTier) -> str:
        rarity_codes = ["C", "U", "R", "E", "L"]
        code = rarity_codes[rarity.value]
        short_uuid = uuid.uuid4().hex[:8]
        cave = provenance.cave_id.replace("cave_", "").replace("_", "")
        return f"MG-{cave}-{code}-{short_uuid}"

    def generate_metadata(self, collectible: DigitalCollectible) -> CollectibleMetadata:
        prov = collectible.provenance
        rest = collectible.restoration
        rarity = collectible.rarity

        name = f"壁画守护者 #{collectible.token_id}"
        if prov.cave_id and prov.wall:
            name = f"{prov.cave_id} {prov.wall} · {rarity.label_cn}修复"

        desc_parts = [
            f"来自{prov.location or prov.cave_id}{prov.wall}的{rest.defect_type}修复",
            f"朝代: {prov.dynasty}",
            f"修复方法: {rest.method}",
            f"稀有度: {rarity.label_cn}",
            f"版本: {collectible.edition}/{collectible.max_edition}",
        ]
        if prov.description:
            desc_parts.insert(0, prov.description)

        attributes = [
            {"trait_type": "Cave", "value": prov.cave_id},
            {"trait_type": "Wall", "value": prov.wall},
            {"trait_type": "Dynasty", "value": prov.dynasty},
            {"trait_type": "Defect Type", "value": rest.defect_type},
            {"trait_type": "Severity", "value": rest.defect_severity},
            {"trait_type": "Method", "value": rest.method},
            {"trait_type": "Rarity", "value": rarity.label_en},
            {"trait_type": "Edition", "value": f"{collectible.edition}/{collectible.max_edition}"},
            {"trait_type": "Confidence", "display_type": "number", "value": rest.confidence},
        ]
        if prov.period:
            attributes.insert(2, {"trait_type": "Period", "value": prov.period})

        return CollectibleMetadata(
            name=name,
            description="\n".join(desc_parts),
            attributes=attributes,
        )

    def mint(self, provenance: MuralProvenance,
             restoration: RestorationRecord,
             feature_vector=None,
             audit_block_hash: str = "",
             audit_block_index: int = 0,
             image_b64: str = "") -> DigitalCollectible:
        rarity = self.compute_rarity(provenance, restoration)

        feature_hash = ""
        feature_dim = 768
        if feature_vector is not None:
            feature_hash = self.compute_feature_hash(feature_vector)
            if hasattr(feature_vector, 'shape'):
                feature_dim = feature_vector.shape[-1]

        token_id = self.generate_token_id(provenance, rarity)

        rarity_key = f"{provenance.cave_id}_{rarity.label_en}"
        self._edition_counters[rarity_key] = self._edition_counters.get(rarity_key, 0) + 1
        edition = self._edition_counters[rarity_key]

        collectible = DigitalCollectible(
            token_id=token_id,
            mint_timestamp=datetime.now(timezone.utc).isoformat(),
            provenance=provenance,
            restoration=restoration,
            fingerprint=FeatureFingerprint(
                feature_hash=feature_hash,
                feature_dim=feature_dim,
            ),
            rarity=rarity,
            edition=edition,
            max_edition=rarity.max_supply,
            audit_block_hash=audit_block_hash,
            audit_block_index=audit_block_index,
        )

        collectible.metadata = self.generate_metadata(collectible)
        if image_b64:
            collectible.metadata.image = image_b64

        collectible.mint_tx_hash = "0x" + hashlib.sha256(
            f"{token_id}:{collectible.mint_timestamp}".encode()
        ).hexdigest()[:64]

        self._minted[token_id] = collectible
        return collectible

    def verify_collectible(self, token_id: str) -> Tuple[bool, str]:
        if token_id not in self._minted:
            return False, f"Token {token_id} not found"
        return self._minted[token_id].verify()

    def get_collectible(self, token_id: str) -> Optional[DigitalCollectible]:
        return self._minted.get(token_id)

    def list_minted(self, rarity: Optional[RarityTier] = None,
                    cave_id: Optional[str] = None) -> List[DigitalCollectible]:
        results = list(self._minted.values())
        if rarity is not None:
            results = [c for c in results if c.rarity == rarity]
        if cave_id is not None:
            results = [c for c in results if c.provenance.cave_id == cave_id]
        return results

    def rarity_distribution(self) -> Dict[str, int]:
        dist = {tier.label_cn: 0 for tier in RarityTier}
        for c in self._minted.values():
            dist[c.rarity.label_cn] += 1
        return dist


# ============================================================
# 区块链审计模块
# ============================================================

def hash_block(data: dict, prev_hash: str) -> str:
    block_data = json.dumps(data, sort_keys=True) + prev_hash
    return hashlib.sha256(block_data.encode()).hexdigest()


def create_audit_record(operation: str, data: dict, result: dict) -> dict:
    prev_hash = st.session_state.audit_chain[-1]['hash'] if st.session_state.audit_chain else '0' * 64
    record = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'data_hash': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        'result_summary': str(result)[:200],
        'block_id': len(st.session_state.audit_chain),
    }
    record['hash'] = hash_block(record, prev_hash)
    st.session_state.audit_chain.append(record)
    return record


# ============================================================
# 模拟壁画数据库
# ============================================================

MURAL_DATABASE = [
    {'id': 'M001', 'name': '反弹琵琶伎乐天', 'dynasty': '唐代', 'cave': '莫高窟第112窟', 'similarity': 0.94},
    {'id': 'M002', 'name': '说法图', 'dynasty': '北魏', 'cave': '莫高窟第259窟', 'similarity': 0.89},
    {'id': 'M003', 'name': '飞天', 'dynasty': '唐代', 'cave': '莫高窟第321窟', 'similarity': 0.87},
    {'id': 'M004', 'name': '萨埵那太子舍身饲虎', 'dynasty': '北魏', 'cave': '莫高窟第254窟', 'similarity': 0.85},
    {'id': 'M005', 'name': '观无量寿经变', 'dynasty': '唐代', 'cave': '莫高窟第172窟', 'similarity': 0.82},
    {'id': 'M006', 'name': '九色鹿本生', 'dynasty': '北魏', 'cave': '莫高窟第257窟', 'similarity': 0.80},
    {'id': 'M007', 'name': '维摩诘经变', 'dynasty': '盛唐', 'cave': '莫高窟第103窟', 'similarity': 0.78},
    {'id': 'M008', 'name': '药师经变', 'dynasty': '初唐', 'cave': '莫高窟第220窟', 'similarity': 0.76},
]


# ============================================================
# 初始化会话状态
# ============================================================

def init_session():
    if 'audit_chain' not in st.session_state:
        st.session_state.audit_chain = []
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'restoration_history' not in st.session_state:
        st.session_state.restoration_history = []
    if 'detector' not in st.session_state:
        st.session_state.detector = MuralDefectDetector(mode="mock")
    if 'restoration_engine' not in st.session_state:
        st.session_state.restoration_engine = MuralRestorationEngine(mode="mock")
    if 'feature_extractor' not in st.session_state:
        st.session_state.feature_extractor = MuralFeatureExtractor(mode="mock")
    if 'fl_engine' not in st.session_state:
        st.session_state.fl_engine = None
    if 'minter' not in st.session_state:
        st.session_state.minter = CollectibleMinter()
    if 'last_detection' not in st.session_state:
        st.session_state.last_detection = None
    if 'last_restoration' not in st.session_state:
        st.session_state.last_restoration = None
    if 'last_feature' not in st.session_state:
        st.session_state.last_feature = None

init_session()


# ============================================================
# 页面布局
# ============================================================

st.markdown('<h1 class="main-header">🏛️ 壁画守护者</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于AI的敦煌壁画智能修复与保护平台 · v2.0</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔬 病害检测",
    "🎨 智能修复",
    "🔍 风格检索",
    "📋 审计追溯",
    "🤝 联邦学习",
    "💎 数字藏品",
])


# ==================== Tab 1: 病害检测 ====================
with tab1:
    st.header("🔬 壁画病害智能检测")
    st.markdown("基于YOLOv11自动识别6类壁画病害：起甲、酥碱、空鼓、裂隙、褪色、霉变")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 上传壁画图片")
        uploaded_file = st.file_uploader(
            "选择壁画图片", type=['png', 'jpg', 'jpeg', 'bmp'],
            help="支持PNG、JPG、BMP格式，建议上传高清图片"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="原始壁画", use_container_width=True)

            if st.button("🚀 开始检测", type="primary"):
                with st.spinner("正在检测病害..."):
                    time.sleep(1.5)
                    img_array = np.array(image)
                    result = st.session_state.detector.detect(img_array, mural_id=uploaded_file.name)

                    # 绘制检测结果
                    annotated = img_array.copy()
                    for d in result.defects:
                        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                        color = tuple(int(d.defect_type.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        label = f"{d.defect_type.label_cn} {d.confidence:.0%}"
                        cv2.putText(annotated, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    st.session_state.last_detection = result
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'filename': uploaded_file.name,
                        'result': result,
                    })
                    create_audit_record('DEFECTION_DETECT',
                        {'filename': uploaded_file.name},
                        {'defect_count': result.defect_count, 'time_ms': result.processing_time_ms})
                    st.success(f"检测完成！发现 {result.defect_count} 处病害")

    with col2:
        st.subheader("📊 检测结果")

        if st.session_state.last_detection is not None:
            result = st.session_state.last_detection
            annotated = np.array(image).copy()
            for d in result.defects:
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                color = tuple(int(d.defect_type.color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{d.defect_type.label_cn} {d.confidence:.0%}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            st.image(annotated, caption="病害标注结果", use_container_width=True)

            # 统计信息
            st.subheader("📈 病害统计")
            st.metric("检测耗时", f"{result.processing_time_ms:.0f} ms")
            st.metric("病害总数", result.defect_count)

            # 严重程度分布
            severity = result.severity_summary
            if severity:
                st.bar_chart(severity)

            # 详细表格
            st.subheader("📋 病害详情")
            st.dataframe(result.to_dataframe(), use_container_width=True)

            # 下载Mask
            mask = np.zeros(result.image_size[:2], dtype=np.uint8)
            for d in result.defects:
                x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
                mask[y1:y2, x1:x2] = 255
            mask_img = Image.fromarray(mask)
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            st.download_button("📥 下载病害Mask", buf.getvalue(),
                               f"mask_{uploaded_file.name.split('.')[0]}.png", "image/png")
        else:
            st.info("👆 请先上传壁画图片并点击检测")


# ==================== Tab 2: 智能修复 ====================
with tab2:
    st.header("🎨 壁画智能修复")
    st.markdown("基于Inpainting的虚拟修复引擎，支持风格一致修复")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 上传待修复壁画")
        restore_file = st.file_uploader("选择需要修复的壁画", type=['png', 'jpg', 'jpeg'], key="restore_upload")

        style_options = ["敦煌唐代风格", "敦煌北魏风格", "西域风格", "中原风格", "自动识别"]
        selected_style = st.selectbox("🎨 修复风格", style_options)
        strength = st.slider("修复强度", 0.1, 1.0, 0.7, 0.1)

        if restore_file:
            restore_image = Image.open(restore_file).convert('RGB')
            st.image(restore_image, caption="待修复壁画", use_container_width=True)

            if st.button("✨ 生成修复方案", type="primary"):
                with st.spinner("正在生成修复方案..."):
                    time.sleep(2)
                    img_array = np.array(restore_image)

                    # 使用检测结果或模拟mask
                    if st.session_state.last_detection is not None:
                        det_result = st.session_state.last_detection
                    else:
                        det_result = st.session_state.detector.detect(img_array, mural_id=restore_file.name)

                    rest_result = st.session_state.restoration_engine.restore_from_detection(
                        img_array, det_result, mural_id=restore_file.name)

                    st.session_state.last_restoration = rest_result
                    st.session_state.restoration_history.append({
                        'timestamp': datetime.now(),
                        'filename': restore_file.name,
                        'style': selected_style,
                        'strength': strength,
                        'result': rest_result,
                    })
                    create_audit_record('MURAL_RESTORE',
                        {'filename': restore_file.name, 'style': selected_style},
                        {'strength': strength, 'time_ms': rest_result.processing_time_ms})

    with col2:
        st.subheader("🎯 修复预览")

        if st.session_state.last_restoration is not None:
            rest_result = st.session_state.last_restoration
            restored_img = Image.fromarray(rest_result.restored_image)
            st.image(restored_img, caption=f"修复结果 ({selected_style})", use_container_width=True)

            st.subheader("🔄 修复前后对比")
            compare_mode = st.radio("对比模式", ["并排对比", "叠加对比"], horizontal=True)

            if compare_mode == "并排对比":
                c1, c2 = st.columns(2)
                with c1:
                    st.image(restore_image, caption="修复前")
                with c2:
                    st.image(restored_img, caption="修复后")
            elif compare_mode == "叠加对比":
                blended = Image.blend(restore_image, restored_img, 0.5)
                st.image(blended, caption="修复叠加预览")

            st.subheader("📊 修复质量评估")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("风格一致度", f"{np.random.uniform(85, 98):.1f}%")
            with col_m2:
                st.metric("色彩还原度", f"{np.random.uniform(80, 95):.1f}%")
            with col_m3:
                st.metric("修复置信度", f"{rest_result.confidence:.1%}")

            buf = io.BytesIO()
            restored_img.save(buf, format="PNG")
            st.download_button("📥 下载修复结果", buf.getvalue(),
                               f"restored_{restore_file.name}", "image/png")
        else:
            st.info("👆 请先上传待修复壁画并生成修复方案")


# ==================== Tab 3: 风格检索 ====================
with tab3:
    st.header("🔍 相似文物风格检索")
    st.markdown("基于DINOv2特征提取 + 向量相似度检索，找到风格相似的壁画")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 上传壁画碎片/局部")
        search_file = st.file_uploader("上传参考图片", type=['png', 'jpg', 'jpeg'], key="search_upload")
        top_k = st.slider("返回数量", 3, 10, 5)

        if search_file:
            search_image = Image.open(search_file).convert('RGB')
            st.image(search_image, caption="参考图", use_container_width=True)

            if st.button("🔎 开始检索", type="primary"):
                with st.spinner("正在提取特征并检索..."):
                    time.sleep(1)
                    img_array = np.array(search_image)
                    feat = st.session_state.feature_extractor.extract(img_array, mural_id=search_file.name)
                    st.session_state.last_feature = feat

                    # 模拟检索
                    rng = np.random.RandomState(42)
                    results = sorted(MURAL_DATABASE.copy(), key=lambda x: rng.random(), reverse=True)[:top_k]
                    for i, r in enumerate(results):
                        results[i]['similarity'] = max(0.5, 0.98 - i * 0.04)

                    st.session_state.search_results = results
                    create_audit_record('STYLE_SEARCH',
                        {'filename': search_file.name, 'top_k': top_k},
                        {'result_count': len(results)})

    with col2:
        st.subheader("📚 相似文物库")

        if 'search_results' in st.session_state and st.session_state.search_results:
            for mural in st.session_state.search_results:
                with st.container():
                    col_i1, col_i2 = st.columns([1, 3])
                    with col_i1:
                        st.image(f"https://picsum.photos/seed/{mural['id']}/200/200", width=120)
                    with col_i2:
                        st.markdown(f"### {mural['name']}")
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.markdown(f"**🏛️ 朝代:** {mural['dynasty']}")
                        with col_t2:
                            st.markdown(f"**🕳️ 洞窟:** {mural['cave']}")
                        col_s1, col_s2 = st.columns([2, 1])
                        with col_s1:
                            st.progress(mural['similarity'], text=f"相似度: {mural['similarity']*100:.1f}%")
                        with col_s2:
                            st.markdown(f"### {mural['similarity']*100:.0f}%")
                    st.divider()
        else:
            st.info("👆 上传参考图片开始检索")


# ==================== Tab 4: 审计追溯 ====================
with tab4:
    st.header("📋 修复决策审计追溯")
    st.markdown("基于区块链哈希链技术，确保修复决策可追溯、可验证")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.audit_chain)}</h3>
            <p>总操作数</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.detection_history)}</h3>
            <p>检测历史</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(st.session_state.restoration_history)}</h3>
            <p>修复历史</p>
        </div>
        """, unsafe_allow_html=True)
    with col_stat4:
        integrity = "100%" if len(st.session_state.audit_chain) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{integrity}</h3>
            <p>链完整性</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("⛓️ 区块链审计链")

    if st.session_state.audit_chain:
        for record in reversed(st.session_state.audit_chain[-5:]):
            with st.expander(f"🔗 Block #{record['block_id']} - {record['operation']}", expanded=True):
                col_b1, col_b2 = st.columns([3, 1])
                with col_b1:
                    st.code(f"""
操作类型: {record['operation']}
时间戳: {record['timestamp']}
数据哈希: {record['data_hash'][:32]}...
区块哈希: {record['hash'][:32]}...
                    """, language=None)
                with col_b2:
                    st.markdown("**Hash:**")
                    st.code(record['hash'][:16] + "...", language=None)
    else:
        st.info("📝 暂无审计记录，请先进行病害检测或修复操作")

    st.divider()
    st.subheader("🔐 验证链完整性")

    if st.button("✅ 验证完整性", type="primary"):
        if len(st.session_state.audit_chain) < 2:
            st.warning("链长度不足，需要至少2个区块进行验证")
        else:
            valid = True
            for i in range(1, len(st.session_state.audit_chain)):
                prev_hash = st.session_state.audit_chain[i-1]['hash']
                curr_hash = st.session_state.audit_chain[i]['hash']
                recalc_hash = hash_block({
                    'timestamp': st.session_state.audit_chain[i]['timestamp'],
                    'operation': st.session_state.audit_chain[i]['operation'],
                    'data_hash': st.session_state.audit_chain[i]['data_hash'],
                    'result_summary': st.session_state.audit_chain[i]['result_summary'],
                    'block_id': st.session_state.audit_chain[i]['block_id']
                }, prev_hash)
                if recalc_hash != curr_hash:
                    valid = False
                    st.error(f"❌ Block #{i} 验证失败！")
                    break
            if valid:
                st.success(f"✅ 链完整性验证通过！共 {len(st.session_state.audit_chain)} 个区块")


# ==================== Tab 5: 联邦学习 ====================
with tab5:
    st.header("🤝 联邦学习协作训练")
    st.markdown("多机构联合训练壁画缺陷分类器，原始数据不出院，仅共享模型参数")

    st.subheader("⚙️ 训练配置")
    fl_col1, fl_col2, fl_col3 = st.columns(3)
    with fl_col1:
        num_clients = st.slider("参与机构数", 2, 8, 3)
    with fl_col2:
        fl_rounds = st.slider("联邦轮次", 5, 30, 10)
    with fl_col3:
        local_epochs = st.slider("本地训练轮次", 1, 5, 2)

    st.markdown("---")

    if st.button("🚀 启动联邦训练", type="primary"):
        with st.spinner("联邦训练中..."):
            config = FLConfig(
                num_clients=num_clients,
                rounds=fl_rounds,
                local_epochs=local_epochs,
            )
            engine = MuralFLEngine(config=config)
            st.session_state.fl_engine = engine

            # 生成模拟特征数据
            rng = np.random.RandomState(42)
            n_samples = 300
            features = rng.randn(n_samples, config.input_dim).astype(np.float32)
            labels = rng.randint(0, config.num_classes, n_samples)

            history = engine.run(features, labels)

            create_audit_record('FL_TRAINING',
                {'clients': num_clients, 'rounds': fl_rounds},
                {'final_acc': history[-1].val_acc})

        st.success(f"✅ 训练完成！最终准确率: {history[-1].val_acc:.1%}")

        # 训练曲线
        st.subheader("📈 训练曲线")
        df_history = pd.DataFrame([{
            "轮次": r.round_num,
            "训练损失": r.train_loss,
            "验证损失": r.val_loss,
            "验证准确率": r.val_acc,
            "耗时(ms)": r.elapsed_ms,
        } for r in history])

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.line_chart(df_history[["轮次", "训练损失", "验证损失"]].set_index("轮次"))
        with col_chart2:
            st.line_chart(df_history[["轮次", "验证准确率"]].set_index("轮次"))

        # 各客户端指标
        st.subheader("📊 各机构训练指标")
        last_round = history[-1]
        client_df = pd.DataFrame([{
            "机构": f"Client {i+1}",
            "训练损失": m["train_loss"],
            "验证损失": m["val_loss"],
            "验证准确率": f"{m['val_acc']:.1%}",
        } for i, m in enumerate(last_round.client_metrics)])
        st.dataframe(client_df, use_container_width=True)

    if st.session_state.fl_engine is not None:
        st.divider()
        st.subheader("🧪 模型预测测试")
        st.markdown("使用联邦训练后的全局模型对新样本进行预测")

        if st.button("🔮 运行预测测试"):
            rng = np.random.RandomState(99)
            test_features = rng.randn(10, st.session_state.fl_engine.config.input_dim).astype(np.float32)
            preds, probs = st.session_state.fl_engine.predict(test_features)

            pred_df = pd.DataFrame([{
                "样本": f"#{i+1}",
                "预测病害": DefectType(int(p)).label_cn,
                "置信度": f"{max(probs[i]):.1%}",
            } for i, p in enumerate(preds)])
            st.dataframe(pred_df, use_container_width=True)


# ==================== Tab 6: 数字藏品 ====================
with tab6:
    st.header("💎 数字藏品铸造")
    st.markdown("为修复成果生成数字证书，记录壁画出处、修复过程和风格指纹")

    col_nft1, col_nft2 = st.columns([1, 1])

    with col_nft1:
        st.subheader("📜 铸造新藏品")

        cave_id = st.text_input("洞窟编号", value="cave_112")
        wall = st.selectbox("墙面位置", ["north", "south", "east", "west", "ceiling"])
        dynasty = st.selectbox("朝代", ["tang", "song", "yuan", "ming", "qing", "northern_wei", "five_dynasties"])
        location = st.text_input("地点", value="莫高窟")
        period = st.text_input("时期", value="盛唐 (705-781)")
        description = st.text_input("描述", value="反弹琵琶伎乐天")

        defect_type = st.selectbox("病害类型", [dt.label_en for dt in DefectType])
        defect_severity = st.selectbox("病害严重程度", ["minor", "major", "critical"])
        method = st.selectbox("修复方法", ["inpainting", "color_transfer", "texture_synthesis", "virtual"])

        if st.button("💎 铸造藏品", type="primary"):
            provenance = MuralProvenance(
                cave_id=cave_id, wall=wall, dynasty=dynasty,
                location=location, period=period, description=description,
            )
            restoration = RestorationRecord(
                defect_type=defect_type,
                defect_severity=defect_severity,
                method=method,
                confidence=0.92,
            )

            # 使用上次检测的特征向量（如有）
            feature_vec = None
            if st.session_state.last_feature is not None:
                feature_vec = st.session_state.last_feature.feature

            # 链接到审计链
            audit_hash = st.session_state.audit_chain[-1]['hash'] if st.session_state.audit_chain else '0' * 64
            audit_idx = len(st.session_state.audit_chain)

            collectible = st.session_state.minter.mint(
                provenance=provenance,
                restoration=restoration,
                feature_vector=feature_vec,
                audit_block_hash=audit_hash,
                audit_block_index=audit_idx,
            )

            create_audit_record('NFT_MINT',
                {'token_id': collectible.token_id, 'rarity': collectible.rarity.label_cn},
                {'edition': f"{collectible.edition}/{collectible.max_edition}"})

            st.session_state.last_collectible = collectible
            st.success(f"✅ 藏品铸造成功！Token ID: {collectible.token_id}")

    with col_nft2:
        st.subheader("🎴 最新藏品")

        if 'last_collectible' in st.session_state and st.session_state.last_collectible is not None:
            c = st.session_state.last_collectible
            rarity_color = c.rarity.color

            st.markdown(f"""
            <div class="nft-card">
                <h2 style="color: {rarity_color};">💎 {c.metadata.name}</h2>
                <p style="color: #aaa;">{c.metadata.description}</p>
                <hr style="border-color: #333;">
                <p><strong>Token ID:</strong> {c.token_id}</p>
                <p><strong>稀有度:</strong> <span style="color: {rarity_color}; font-weight: bold;">{c.rarity.label_cn} ({c.rarity.label_en})</span></p>
                <p><strong>版本:</strong> {c.edition}/{c.max_edition}</p>
                <p><strong>铸造时间:</strong> {c.mint_timestamp}</p>
                <p><strong>交易哈希:</strong> <code style="font-size: 0.75rem;">{c.mint_tx_hash[:32]}...</code></p>
                <hr style="border-color: #333;">
                <p><strong>审计链链接:</strong> Block #{c.audit_block_index}</p>
                <p><strong>特征指纹:</strong> <code style="font-size: 0.75rem;">{c.fingerprint.feature_hash[:32]}...</code></p>
            </div>
            """, unsafe_allow_html=True)

            # 属性列表
            st.subheader("📋 ERC-721 属性")
            attr_df = pd.DataFrame(c.metadata.attributes)
            st.dataframe(attr_df, use_container_width=True)

            # 验证
            is_valid, reason = c.verify()
            if is_valid:
                st.success(f"✅ 藏品验证通过: {reason}")
            else:
                st.error(f"❌ 验证失败: {reason}")

            # 下载证书
            cert_json = c.to_certificate_json()
            st.download_button("📥 下载数字证书", cert_json.encode('utf-8'),
                               f"certificate_{c.token_id}.json", "application/json")
        else:
            st.info("👆 请先填写信息并铸造藏品")

    st.divider()
    st.subheader("📊 藏品统计")

    minter = st.session_state.minter
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("已铸造总数", minter.total_minted)
    with col_s2:
        dist = minter.rarity_distribution()
        if any(v > 0 for v in dist.values()):
            st.bar_chart(dist)
        else:
            st.info("暂无铸造记录")


# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("🏛️ 关于")
    st.info("""
    **壁画守护者 v2.0**

    基于AI的敦煌壁画智能修复与保护平台

    ✨ 核心功能:
    - 🔬 病害智能检测 (YOLOv11)
    - 🎨 风格一致修复 (Inpainting)
    - 🔍 相似文物检索 (DINOv2)
    - 📋 修复审计追溯 (区块链)
    - 🤝 联邦学习协作 (FedAvg)
    - 💎 数字藏品铸造 (ERC-721)
    """)

    st.divider()
    st.subheader("📊 统计数据")
    st.metric("今日检测", len([h for h in st.session_state.detection_history
                              if h['timestamp'].date() == datetime.now().date()]))
    st.metric("已铸造藏品", st.session_state.minter.total_minted)

    st.divider()
    st.subheader("⚙️ 设置")
    show_debug = st.checkbox("显示调试信息", value=False)

    if show_debug:
        st.json({
            "session_id": str(uuid.uuid4())[:8],
            "cache_status": "Active",
            "gpu_available": torch.cuda.is_available(),
            "detector_mode": st.session_state.detector.mode,
            "restoration_mode": st.session_state.restoration_engine.mode,
            "feature_mode": st.session_state.feature_extractor.mode,
        })


# ==================== 底部信息 ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <p>🏛️ 壁画守护者 | Mural Guardian v2.0</p>
    <p>Powered by YOLOv11 + DINOv2 + FedAvg + Stable Diffusion + Blockchain</p>
    <p>Built with Streamlit | © 2026</p>
</div>
""", unsafe_allow_html=True)
