# 🏛️ 壁画守护者 (Mural Guardian)

> 基于AI的敦煌壁画智能修复与保护平台

![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ 核心功能

| 功能 | 描述 | 技术 |
|------|------|------|
| 🔬 **病害智能检测** | 自动识别壁画病害（起甲、酥碱、空鼓等） | SAM + UNet |
| 🎨 **风格一致修复** | 基于同期文物生成风格匹配的修复方案 | Stable Diffusion + ControlNet |
| 🔍 **相似文物检索** | 基于HNSW向量检索找到相似壁画图谱 | ResNet-18 + Rust HNSW |
| 📋 **修复审计追溯** | 区块链哈希链确保修复决策可追溯 | SHA-256 Hash Chain |

## 🚀 快速开始

### 本地运行

```bash
# 克隆项目
git clone https://github.com/dechang64/mural-restoration.git
cd mural-restoration

# 安装依赖
pip install -r requirements.txt

# 下载SAM模型权重 (可选)
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 运行应用
streamlit run app.py
```

### Streamlit Cloud 部署

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_github.svg)](https://share.streamlit.io/)

1. Fork 此仓库
2. 访问 [share.streamlit.io](https://share.streamlit.io/)
3. 选择你的仓库和 `app.py`
4. 部署完成！

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         前端展示层 (Streamlit)                      │
│   病害标注 | 修复预览 | 风格检索 | 审计追溯                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────┐
│                      API 网关层 (FastAPI)                          │
│   /detect    /restore   /search   /audit                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  病害检测服务  │    │  修复生成服务  │    │  向量检索服务  │
│  SAM + UNet   │    │  ControlNet   │    │  HNSW (Rust)  │
│  PyTorch      │    │  + LoRA       │    │  gRPC         │
└───────────────┘    └───────────────┘    └───────────────┘
```

## 📁 项目结构

```
mural-restoration/
├── app.py                 # Streamlit 主应用
├── requirements.txt       # Python 依赖
├── PROPOSAL.md           # 项目提案文档
├── README.md             # 项目说明
├── models/               # 模型权重 (可选)
│   └── sam_vit_b_01ec64.pth
├── data/                 # 数据目录
│   ├── raw/             # 原始壁画图片
│   ├── masks/           # 病害标注
│   └── results/         # 修复结果
└── utils/               # 工具函数
    ├── detection.py     # 病害检测
    ├── restoration.py   # 修复生成
    ├── search.py        # 风格检索
    └── audit.py         # 审计追溯
```

## 🔬 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 前端 | Streamlit | 快速构建数据应用 |
| 病害检测 | SAM (Segment Anything) | Meta 开源的分割模型 |
| 图像生成 | Stable Diffusion | 扩散模型生成修复方案 |
| 特征提取 | ResNet-18 / ViT | 图像特征向量化 |
| 向量检索 | HNSW | 高效近似最近邻检索 |
| 联邦学习 | PyTorch + gRPC | 多机构协同训练 |
| 区块链审计 | SHA-256 Hash Chain | 操作不可篡改 |

## 📊 预期效果

- 🔬 病害识别准确率: > 90%
- ⚡ 修复方案生成时间: < 30秒/图
- 🔍 风格检索召回率: > 85%
- ⛓️ 区块链存证: 全流程可审计

## 🗺️ 发展路线

- [ ] **Phase 1**: MVP - Streamlit 单机版
- [ ] **Phase 2**: 联邦学习版 - 多机构协作
- [ ] **Phase 3**: 生产级 - API + 移动端AR

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

---

🏛️ **壁画守护者** - 用AI守护千年文化遗产
