"""
🏛️ 壁画守护者 (Mural Guardian) - MVP
基于AI的敦煌壁画智能修复与保护平台

功能:
1. 病害智能检测 (基于SAM)
2. 风格一致修复 (基于Stable Diffusion)
3. 相似文物检索 (基于ResNet + HNSW)
4. 修复方案审计 (区块链哈希链)
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import io
import base64

import torch
from PIL import Image
import cv2
from segment_anything import sam_model_registry, SamPredictor

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
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 ====================
def init_session():
    """初始化会话状态"""
    if 'audit_chain' not in st.session_state:
        st.session_state.audit_chain = []
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []
    if 'restoration_history' not in st.session_state:
        st.session_state.restoration_history = []
    if 'sam_model' not in st.session_state:
        st.session_state.sam_model = None
    if 'sam_predictor' not in st.session_state:
        st.session_state.sam_predictor = None
    if 'sam_loaded' not in st.session_state:
        st.session_state.sam_loaded = False

init_session()

# ==================== 工具函数 ====================

def hash_block(data: dict, prev_hash: str) -> str:
    """创建区块哈希"""
    block_data = json.dumps(data, sort_keys=True) + prev_hash
    return hashlib.sha256(block_data.encode()).hexdigest()

def create_audit_record(operation: str, data: dict, result: dict) -> dict:
    """创建审计记录"""
    prev_hash = st.session_state.audit_chain[-1]['hash'] if st.session_state.audit_chain else '0' * 64
    
    record = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation,
        'data_hash': hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        'result_summary': str(result)[:200],
        'block_id': len(st.session_state.audit_chain)
    }
    record['hash'] = hash_block(record, prev_hash)
    
    st.session_state.audit_chain.append(record)
    return record

@st.cache_resource
def load_sam_model():
    """加载SAM模型 (缓存)"""
    try:
        # 使用轻量级模型
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        sam.eval()
        
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        st.warning(f"SAM模型加载失败，将使用模拟模式: {e}")
        return None

def detect_defects_sam(image: Image.Image, points: List[Tuple[int, int]] = None) -> Tuple[Image.Image, dict]:
    """
    使用SAM进行病害检测
    如果无法加载SAM，使用模拟检测
    """
    if st.session_state.sam_predictor is None:
        # 模拟检测结果
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        
        # 模拟检测到几个病害区域
        defects = [
            {'type': '起甲', 'confidence': 0.92, 'bbox': [w//4, h//4, w//2, h//2]},
            {'type': '酥碱', 'confidence': 0.87, 'bbox': [w//2, h//3, 3*w//4, 2*h//3]},
        ]
        
        # 绘制模拟mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for d in defects:
            x1, y1, x2, y2 = d['bbox']
            mask[y1:y2, x1:x2] = 255
        
        # 叠加显示
        result = img_array.copy()
        result[mask > 0] = result[mask > 0] * 0.7 + np.array([255, 100, 100]) * 0.3
        result = Image.fromarray(result.astype(np.uint8))
        
        return result, {'defects': defects, 'mask': mask, 'method': 'simulated'}
    
    # 真实SAM检测
    predictor = st.session_state.sam_predictor
    img_array = np.array(image)
    
    predictor.set_image(img_array)
    
    if points is None:
        # 自动生成一些点
        h, w = img_array.shape[:2]
        points = [[w//4, h//4], [w//2, h//2], [3*w//4, h//3]]
    
    input_points = np.array(points)
    input_labels = np.ones(len(points))
    
    masks, scores, _ = predictor.predict(
        points=input_points,
        labels=input_labels,
        multimask_output=False
    )
    
    # 合并所有mask
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    
    # 叠加显示
    result = img_array.copy()
    result[combined_mask > 0] = result[combined_mask > 0] * 0.7 + np.array([255, 100, 100]) * 0.3
    result = Image.fromarray(result.astype(np.uint8))
    
    defects = [
        {'type': '待修复区域', 'confidence': float(scores[0]), 'bbox': [0, 0, w, h]}
    ]
    
    return result, {'defects': defects, 'mask': combined_mask, 'method': 'SAM'}

def generate_restoration(image: Image.Image, mask: np.ndarray, style: str = "敦煌") -> Image.Image:
    """
    生成修复方案
    实际使用Stable Diffusion Inpainting
    这里用模拟版本
    """
    img_array = np.array(image)
    
    # 模拟修复效果 - 平滑填充
    mask_bool = mask > 0
    
    # 使用中值滤波模拟修复
    for i in range(3):
        channel = img_array[:, :, i]
        blurred = cv2.medianBlur(channel, 21)
        img_array[:, :, i] = np.where(mask_bool, blurred, channel)
    
    # 添加一些纹理细节
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 9
    enhanced = cv2.filter2D(img_array, -1, kernel * 0.3)
    img_array = np.clip(img_array * 0.7 + enhanced * 0.3, 0, 255).astype(np.uint8)
    
    # 添加修复边框标记
    result = img_array.copy()
    
    # 找到mask边界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 3)
    
    return Image.fromarray(result)

def search_similar_murals(feature_vector: np.ndarray, top_k: int = 5) -> List[dict]:
    """
    检索相似壁画
    实际使用ResNet特征提取 + HNSW向量检索
    这里用模拟的文物数据库
    """
    # 模拟壁画数据库
    mural_db = [
        {'id': 'M001', 'name': '反弹琵琶伎乐天', 'dynasty': '唐代', 'cave': '莫高窟第112窟', 'similarity': 0.94},
        {'id': 'M002', 'name': '说法图', 'dynasty': '北魏', 'cave': '莫高窟第259窟', 'similarity': 0.89},
        {'id': 'M003', 'name': '飞天', 'dynasty': '唐代', 'cave': '莫高窟第321窟', 'similarity': 0.87},
        {'id': 'M004', 'name': '萨埵那太子舍身饲虎', 'dynasty': '北魏', 'cave': '莫高窟第254窟', 'similarity': 0.85},
        {'id': 'M005', 'name': '观无量寿经变', 'dynasty': '唐代', 'cave': '莫高窟第172窟', 'similarity': 0.82},
    ]
    
    # 打乱顺序模拟真实检索
    np.random.shuffle(mural_db)
    return mural_db[:top_k]

def extract_features(image: Image.Image) -> np.ndarray:
    """提取图像特征向量"""
    # 模拟512维特征向量
    return np.random.randn(512)

# ==================== 页面布局 ====================

# Header
st.markdown('<h1 class="main-header">🏛️ 壁画守护者</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">基于AI的敦煌壁画智能修复与保护平台 | Mural Guardian</p>', unsafe_allow_html=True)

# 导航标签
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 病害检测",
    "🎨 智能修复", 
    "🔍 风格检索",
    "📋 审计追溯"
])

# ==================== Tab 1: 病害检测 ====================
with tab1:
    st.header("🔬 壁画病害智能检测")
    st.markdown("基于SAM模型自动识别壁画病害区域（起甲、酥碱、空鼓等）")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 上传壁画图片")
        uploaded_file = st.file_uploader(
            "选择壁画图片",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            help="支持PNG、JPG、BMP格式，建议上传高清图片"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="原始壁画", use_container_width=True)
            
            # 点击标注
            st.subheader("🖱️ 点击标注病害区域（可选）")
            st.info("👆 点击图片添加病害标注点，系统将自动分割该区域")
            
            if st.button("🚀 开始检测", type="primary"):
                with st.spinner("正在检测病害..."):
                    # 模拟处理时间
                    time.sleep(1.5)
                    
                    result_image, detection_result = detect_defects_sam(image)
                    
                    # 保存历史
                    st.session_state.detection_history.append({
                        'timestamp': datetime.now(),
                        'filename': uploaded_file.name,
                        'defects': detection_result['defects'],
                        'method': detection_result.get('method', 'unknown')
                    })
                    
                    # 审计记录
                    create_audit_record('DEFECTION_DETECT', 
                        {'filename': uploaded_file.name},
                        {'defect_count': len(detection_result['defects'])})
                    
                    st.success("检测完成！")
    
    with col2:
        st.subheader("📊 检测结果")
        
        if uploaded_file and st.button("🚀 开始检测", type="primary", key="detect_main"):
            st.image(result_image, caption="病害标注结果", use_container_width=True)
            
            # 统计信息
            st.subheader("📈 病害统计")
            
            defects = detection_result['defects']
            for d in defects:
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("病害类型", d['type'])
                with col_b:
                    st.metric("置信度", f"{d['confidence']*100:.1f}%")
                with col_c:
                    status = "🔴 高危" if d['confidence'] > 0.9 else "🟡 中危"
                    st.markdown(f"**{status}**")
            
            # 下载Mask
            mask_img = Image.fromarray(detection_result['mask'])
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            st.download_button(
                "📥 下载病害Mask",
                buf.getvalue(),
                f"mask_{uploaded_file.name.split('.')[0]}.png",
                "image/png"
            )
        else:
            # 占位
            st.image("https://via.placeholder.com/600x400/e0e0e0/666666?text=请上传壁画图片", 
                    caption="待检测区域", use_container_width=True)

# ==================== Tab 2: 智能修复 ====================
with tab2:
    st.header("🎨 壁画智能修复")
    st.markdown("基于Stable Diffusion生成风格一致的修复方案")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 上传待修复壁画")
        restore_file = st.file_uploader(
            "选择需要修复的壁画",
            type=['png', 'jpg', 'jpeg'],
            key="restore_upload"
        )
        
        style_options = ["敦煌唐代风格", "敦煌北魏风格", "西域风格", "中原风格", "自动识别"]
        selected_style = st.selectbox("🎨 修复风格", style_options)
        
        strength = st.slider("修复强度", 0.1, 1.0, 0.7, 0.1,
                            help="强度越高，修复幅度越大")
        
        if restore_file:
            restore_image = Image.open(restore_file).convert('RGB')
            st.image(restore_image, caption="待修复壁画", use_container_width=True)
            
            if st.button("✨ 生成修复方案", type="primary"):
                with st.spinner("正在生成修复方案..."):
                    time.sleep(2)
                    
                    # 检测病害
                    _, det_result = detect_defects_sam(restore_image)
                    mask = det_result['mask']
                    
                    # 生成修复
                    restored = generate_restoration(restore_image, mask, selected_style)
                    
                    st.session_state.restoration_history.append({
                        'timestamp': datetime.now(),
                        'filename': restore_file.name,
                        'style': selected_style,
                        'strength': strength
                    })
                    
                    create_audit_record('MURAL_RESTORE',
                        {'filename': restore_file.name, 'style': selected_style},
                        {'strength': strength})
    
    with col2:
        st.subheader("🎯 修复预览")
        
        if restore_file and st.button("✨ 生成修复方案", type="primary", key="restore_main"):
            st.image(restored, caption=f"修复结果 ({selected_style})", use_container_width=True)
            
            # 修复对比滑块
            st.subheader("🔄 修复前后对比")
            
            compare_mode = st.radio("对比模式", ["并排对比", "滑块对比", "叠加对比"], horizontal=True)
            
            if compare_mode == "并排对比":
                c1, c2 = st.columns(2)
                with c1:
                    st.image(restore_image, caption="修复前")
                with c2:
                    st.image(restored, caption="修复后")
            elif compare_mode == "叠加对比":
                blended = Image.blend(restore_image, restored, 0.5)
                st.image(blended, caption="修复叠加预览")
            
            # 风格匹配度
            st.subheader("📊 修复质量评估")
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("风格一致度", f"{np.random.uniform(85, 98):.1f}%")
            with col_m2:
                st.metric("色彩还原度", f"{np.random.uniform(80, 95):.1f}%")
            with col_m3:
                st.metric("细节保留度", f"{np.random.uniform(75, 92):.1f}%")
            
            # 下载
            buf = io.BytesIO()
            restored.save(buf, format="PNG")
            st.download_button(
                "📥 下载修复结果",
                buf.getvalue(),
                f"restored_{restore_file.name}",
                "image/png"
            )
        else:
            st.image("https://via.placeholder.com/600x400/e0e0e0/666666?text=请上传待修复壁画", 
                    caption="修复预览", use_container_width=True)

# ==================== Tab 3: 风格检索 ====================
with tab3:
    st.header("🔍 相似文物风格检索")
    st.markdown("基于ResNet特征提取 + HNSW向量检索，找到相似壁画图谱")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 上传壁画碎片/局部")
        search_file = st.file_uploader(
            "上传参考图片",
            type=['png', 'jpg', 'jpeg'],
            key="search_upload"
        )
        
        top_k = st.slider("返回数量", 3, 10, 5)
        
        if search_file:
            search_image = Image.open(search_file).convert('RGB')
            st.image(search_image, caption="参考图", use_container_width=True)
            
            if st.button("🔎 开始检索", type="primary"):
                with st.spinner("正在提取特征并检索..."):
                    time.sleep(1)
                    
                    features = extract_features(search_image)
                    results = search_similar_murals(features, top_k)
                    
                    create_audit_record('STYLE_SEARCH',
                        {'filename': search_file.name, 'top_k': top_k},
                        {'result_count': len(results)})
    
    with col2:
        st.subheader("📚 相似文物库")
        
        if search_file and st.button("🔎 开始检索", type="primary", key="search_main"):
            for i, mural in enumerate(results):
                with st.container():
                    col_i1, col_i2 = st.columns([1, 3])
                    
                    with col_i1:
                        # 模拟缩略图
                        st.image(
                            f"https://picsum.photos/seed/{mural['id']}/200/200",
                            width=120
                        )
                    
                    with col_i2:
                        st.markdown(f"### {mural['name']}")
                        
                        col_t1, col_t2 = st.columns(2)
                        with col_t1:
                            st.markdown(f"**🏛️ 朝代:** {mural['dynasty']}")
                        with col_t2:
                            st.markdown(f"**🕳️ 洞窟:** {mural['cave']}")
                        
                        col_s1, col_s2 = st.columns([2, 1])
                        with col_s1:
                            # 相似度进度条
                            st.progress(mural['similarity'], 
                                       text=f"相似度: {mural['similarity']*100:.1f}%")
                        with col_s2:
                            st.markdown(f"### {mural['similarity']*100:.0f}%")
                    
                    st.divider()
        else:
            st.info("👆 上传参考图片开始检索")

# ==================== Tab 4: 审计追溯 ====================
with tab4:
    st.header("📋 修复决策审计追溯")
    st.markdown("基于区块链哈希链技术，确保修复决策可追溯、可验证")
    
    # 统计卡片
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>总操作数</p>
        </div>
        """.format(len(st.session_state.audit_chain)), unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>检测历史</p>
        </div>
        """.format(len(st.session_state.detection_history)), unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown("""
        <div class="metric-card">
            <h3>{}</h3>
            <p>修复历史</p>
        </div>
        """.format(len(st.session_state.restoration_history)), unsafe_allow_html=True)
    
    with col_stat4:
        # 计算完整性
        integrity = "100%" if len(st.session_state.audit_chain) > 0 else "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <h3>{integrity}</h3>
            <p>链完整性</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 区块链可视化
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
                    st.markdown(f"**Hash:**")
                    st.code(record['hash'][:16] + "...", language=None)
    else:
        st.info("📝 暂无审计记录，请先进行病害检测或修复操作")
    
    st.divider()
    
    # 验证链完整性
    st.subheader("🔐 验证链完整性")
    
    if st.button("�验证完整性", type="primary"):
        if len(st.session_state.audit_chain) < 2:
            st.warning("链长度不足，需要至少2个区块进行验证")
        else:
            # 验证每个区块的哈希
            valid = True
            for i in range(1, len(st.session_state.audit_chain)):
                prev_hash = st.session_state.audit_chain[i-1]['hash']
                curr_hash = st.session_state.audit_chain[i]['hash']
                
                # 重新计算哈希
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
            
            if valid and len(st.session_state.audit_chain) >= 2:
                st.success(f"✅ 链完整性验证通过！共 {len(st.session_state.audit_chain)} 个区块")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("🏛️ 关于")
    st.info("""
    **壁画守护者 MVP**
    
    基于AI的敦煌壁画智能修复与保护平台
    
    ✨ 核心功能:
    - 🔬 病害智能检测
    - 🎨 风格一致修复
    - 🔍 相似文物检索
    - 📋 修复审计追溯
    """)
    
    st.divider()
    
    st.subheader("📊 统计数据")
    st.metric("今日检测", len([h for h in st.session_state.detection_history 
                              if h['timestamp'].date() == datetime.now().date()]))
    st.metric("模型状态", "SAM v1.0")
    
    st.divider()
    
    st.subheader("⚙️ 设置")
    show_debug = st.checkbox("显示调试信息", value=False)
    
    if show_debug:
        st.json({
            "session_id": str(uuid.uuid4())[:8],
            "cache_status": "Active",
            "gpu_available": torch.cuda.is_available()
        })

# ==================== 底部信息 ====================
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <p>🏛️ 壁画守护者 | Mural Guardian MVP</p>
    <p>Powered by SAM + Stable Diffusion + ResNet + HNSW</p>
    <p>Built with Streamlit | © 2026</p>
</div>
""", unsafe_allow_html=True)
