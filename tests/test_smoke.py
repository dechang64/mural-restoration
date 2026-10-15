# -*- coding: utf-8 -*-
"""Mural Guardian 冒烟测试
2026-10-16 审计后建立。运行: python -m pytest tests/ -v  (或直接 python tests/test_smoke.py)

注: 测试通过 stub streamlit 导入 app.py，对全部用户可见数值/演示数据做诚实性守卫。
未来若有人把指标改回 np.random 掷骰子，这里的测试会当场翻脸。
"""
import os
import sys
import types
import hashlib
import io as _io

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------- streamlit stub ----------------
class _DummyCtx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __call__(self, *a, **k):
        return _DummyCtx()
    def __getattr__(self, name):
        return lambda *a, **k: _DummyCtx() if name in (
            'expander', 'container', 'columns', 'tabs', 'spinner', 'form') else None


class _SessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        self[k] = v


def _install_streamlit_stub():
    st = types.ModuleType("streamlit")
    st.session_state = _SessionState()
    st.set_page_config = lambda *a, **k: None
    st.markdown = lambda *a, **k: None
    st.header = st.subheader = st.title = st.write = st.caption = st.code = lambda *a, **k: None
    st.metric = st.info = st.warning = st.success = st.error = st.divider = lambda *a, **k: None
    st.image = st.dataframe = st.bar_chart = st.line_chart = st.progress = lambda *a, **k: None
    st.button = lambda *a, **k: False
    st.checkbox = lambda *a, **k: False
    st.file_uploader = lambda *a, **k: None
    st.selectbox = lambda *a, **k: (k.get('options') or [''])[0]
    st.radio = lambda *a, **k: (k.get('options') or [''])[0]
    st.slider = lambda *a, **k: k.get('value', 0)
    st.download_button = lambda *a, **k: None
    st.text_input = lambda *a, **k: k.get('value', '')
    st.text_area = lambda *a, **k: k.get('value', '')
    st.number_input = lambda *a, **k: k.get('value', 0)
    st.multiselect = lambda *a, **k: []
    def _st_getattr(name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return lambda *a, **k: None
    st.__file__ = '<streamlit-stub>'
    st.__getattr__ = _st_getattr
    st.sidebar = _DummyCtx()
    st.tabs = lambda labels: [_DummyCtx() for _ in labels]
    def _cols(*a, **k):
        spec = a[0] if a else 2
        n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_DummyCtx() for _ in range(n)]
    st.columns = _cols
    st.expander = lambda *a, **k: _DummyCtx()
    st.container = lambda *a, **k: _DummyCtx()
    st.spinner = lambda *a, **k: _DummyCtx()
    sys.modules["streamlit"] = st
    return st


_st = _install_streamlit_stub()

import app  # noqa: E402  (依赖stub，必须在stub之后导入)


# ---------------- 测试 ----------------
def test_seed_determinism():
    """同一 mural_id 的模拟检测结果必须可复现(审计修正: 原 hash() 受 PYTHONHASHSEED 影响)"""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    d1 = app.MuralDefectDetector(mode="mock")
    d2 = app.MuralDefectDetector(mode="mock")
    r1 = d1.detect(img, mural_id="cave_257_test")
    r2 = d2.detect(img, mural_id="cave_257_test")
    assert r1.defect_count == r2.defect_count
    assert [(d.x1, d.y1, d.class_id) for d in r1.defects] == \
           [(d.x1, d.y1, d.class_id) for d in r2.defects]


def test_defect_taxonomy_six_classes():
    for dt in app.DefectType:
        assert dt.label_cn and dt.label_en and dt.severity and dt.color
    assert len(list(app.DefectType)) == 6


def test_yolo_requires_custom_weights():
    """无病害权重时必须降级 mock 并记录原因(审计修正: 禁止 COCO 权重 %6 硬映射)"""
    det = app.MuralDefectDetector(mode="yolo")
    assert det.mode == "mock"
    assert det._fallback_reason


def test_yolo_class_filter_never_maps_coco():
    """源码级守卫: 不允许出现对检测类别取模映射的行"""
    src = _io.open(app.__file__, encoding='utf-8').read()
    assert 'int(box.cls[0]) % 6' not in src


def test_no_random_user_metrics():
    """源码级守卫: 用户可见指标禁止 np.random(审计修正: 掷骰子事件不得重演)"""
    src = _io.open(app.__file__, encoding='utf-8').read()
    assert 'np.random.uniform(85, 98)' not in src
    assert 'np.random.uniform(80, 95)' not in src


def test_no_external_placeholder_images():
    """源码级守卫: 禁止外链占位图(picsum)，与"数据不出机构"叙事一致"""
    src = _io.open(app.__file__, encoding='utf-8').read()
    assert 'https://picsum.photos' not in src


def test_dunhuang_demo_db():
    assert len(app.MURAL_DATABASE) >= 6
    for m in app.MURAL_DATABASE:
        assert '演示数据' in m['name'], f"{m['name']} 必须标注演示数据"
        assert '莫高窟' in m['cave']


def test_audit_chain_integrity():
    """哈希链: 篡改任一块，验证必失败"""
    _st.session_state.audit_chain = []
    app.create_audit_record('OP_A', {'k': 1}, {'ok': True})
    app.create_audit_record('OP_B', {'k': 2}, {'ok': True})
    app.create_audit_record('OP_C', {'k': 3}, {'ok': True})
    chain = _st.session_state.audit_chain
    assert len(chain) == 3
    for i in range(1, len(chain)):
        recalc = app.hash_block({
            'timestamp': chain[i]['timestamp'],
            'operation': chain[i]['operation'],
            'data_hash': chain[i]['data_hash'],
            'result_summary': chain[i]['result_summary'],
            'block_id': chain[i]['block_id'],
        }, chain[i - 1]['hash'])
        assert recalc == chain[i]['hash']
    # 篡改检测: 改动任一字段后，该块自身哈希必然失配(内容承诺 + 前块链接双保险)
    chain[1]['operation'] = 'TAMPERED'
    tampered_recalc = app.hash_block({
        'timestamp': chain[1]['timestamp'],
        'operation': chain[1]['operation'],
        'data_hash': chain[1]['data_hash'],
        'result_summary': chain[1]['result_summary'],
        'block_id': chain[1]['block_id'],
    }, chain[0]['hash'])
    assert tampered_recalc != chain[1]['hash']


def test_rarity_bounds():
    minter = app.CollectibleMinter()
    low = minter.compute_rarity(
        app.MuralProvenance(dynasty="modern"),
        app.RestorationRecord(defect_type="fading", defect_severity="minor"))
    assert low == app.RarityTier.COMMON
    high = minter.compute_rarity(
        app.MuralProvenance(dynasty="sixteen_kingdoms"),
        app.RestorationRecord(defect_type="hollowing", defect_severity="critical"))
    assert high == app.RarityTier.LEGENDARY


def test_token_hash_stable_and_sensitive():
    c = app.DigitalCollectible(token_id="MG-test", mint_timestamp="t")
    h1 = c.compute_token_hash()
    assert h1 == c.compute_token_hash()
    c.edition = 2
    assert c.compute_token_hash() != h1
    assert len(h1) == 64


def test_nft_flag_default_off():
    assert app.ENABLE_NFT is False


def test_mock_confidence_not_hardcoded():
    """演示模式的修复结果不得携带伪造置信度"""
    eng = app.MuralRestorationEngine(mode="mock")
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    res = eng.restore(img, mask, mural_id="t")
    assert res.confidence == 0.0


def test_placeholder_image_local():
    a = app._placeholder_image("M001")
    b = app._placeholder_image("M001")
    c = app._placeholder_image("M002")
    assert a.size == (200, 200)
    assert np.array(a).tobytes() == np.array(b).tobytes()
    assert np.array(a).tobytes() != np.array(c).tobytes()


def test_restoration_result_shapes():
    eng = app.MuralRestorationEngine(mode="mock")
    img = (np.random.RandomState(0).rand(40, 40, 3) * 255).astype(np.uint8)
    det = app.DetectionResult(mural_id="t", image_size=(40, 40), defects=[
        app.DefectBox(5, 5, 15, 15, 0.9, 2)])
    res = eng.restore_from_detection(img, det)
    assert res.restored_image.shape == img.shape
    assert res.defect_mask.sum() > 0


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f'PASS {fn.__name__}')
        passed += 1
    print(f'{passed} passed')
