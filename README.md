<div align="center">

# 🏛️ Mural Guardian v2

### 联邦学习驱动的壁画智能修复与保护平台

**壁画数据不出机构 — 但修复智慧在社群间流动**

[![Rust](https://img.shields.io/badge/Rust-1.70+-orange?logo=rust)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-9b59b6)](https://docs.ultralytics.com/)
[![DINOv2](https://img.shields.io/badge/DINOv2-Meta-blueviolet)](https://github.com/facebookresearch/dinov2)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

---

## 🎯 Problem

以敦煌为例，壁画遗址拥有 **4.5万㎡壁画**，但仅有 **100多名修复师**：
- **数据孤岛**: 各机构壁画数据受文物保护法规限制，无法共享
- **修复一致性**: 修复师个人风格差异大，同期文物风格不统一
- **病害发现滞后**: 依赖专家肉眼识别，微小病害难以及时发现

## 🆕 v2 Features

| Feature | v1 (MVP) | v2 |
|---------|----------|----|
| Defect Detection | SAM (mock) | **YOLOv11** (6 defect types) |
| Feature Extraction | — | **DINOv2** (768-dim style features) |
| Style Search | Hardcoded list | **HNSW vector search** (Rust) |
| Virtual Restoration | Median filter | **Diffusion inpainting** |
| FL Engine | — | **FedAvg** (multi-institution) |
| Backend | — | **Rust** (HNSW + gRPC + audit chain) |
| Audit | SHA-256 in-memory | **Blockchain hash chain** (Rust) |
| Dashboard | Streamlit | **Axum web dashboard** |

## 🏥 Defect Types

| # | Type (EN) | 中文 | Severity |
|---|-----------|------|----------|
| 0 | Flaking | 起甲 | 🟡 Major |
| 1 | Saline | 酥碱 | 🔴 Critical |
| 2 | Hollowing | 空鼓 | 🔴 Critical |
| 3 | Cracking | 裂隙 | 🟡 Major |
| 4 | Fading | 褪色 | 🟢 Minor |
| 5 | Mold | 霉变 | 🟡 Major |

## 🚀 Quick Start

```bash
# Server (Rust)
cargo run

# Client (Python)
cd python && pip install -r requirements.txt
python -c "
from analysis import MuralFeatureExtractor, MuralDefectDetector
import numpy as np

# Feature extraction
ext = MuralFeatureExtractor(mode='mock', dim=768)
img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
feat = ext.extract(img, mural_id='cave_45_north')
print(f'Feature dim: {feat.dim}')

# Defect detection
det = MuralDefectDetector(mode='mock')
result = det.detect(img, mural_id='cave_45')
print(f'Defects found: {result.num_defects}')
print(f'Health score: {result.health_score:.1f}')
"
```

## 📊 Tests

```bash
cd python && python -m pytest tests/ -v
# 42 passed
```

## 🤝 Related Projects

| Project | Domain | Shared Infra |
|---------|--------|-------------|
| [organoid-fl](https://github.com/dechang64/organoid-fl) | Medical imaging | YOLOv11, DINOv2, SAM2, Grad-CAM |
| [embodied-fl](https://github.com/dechang64/embodied-fl) | Robotics | DINOv2, Multi-Task FL |
| [defect-fl](https://github.com/dechang64/defect-fl) | PCB inspection | YOLOv11, DINOv2, FedAvg |
| [FundFL](https://github.com/dechang64/FundFL) | Finance | HNSW, audit chain |
| [Reading-FL](https://github.com/dechang64/reading-fl) | Reading | DINOv2, FedAvg |

## 📄 License

Apache-2.0

---

<div align="center">

**Mural Guardian v2** — 用AI守护千年文化遗产

</div>
