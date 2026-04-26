//! Web dashboard for the Mural Guardian VectorDB platform.
//!
//! Provides a REST API and embedded HTML dashboard for monitoring:
//! - Mural feature vector statistics
//! - Similar mural search interface
//! - Restoration audit log viewer
//! - Blockchain integrity verification

use crate::blockchain::{AuditChain, OperationType};
use crate::db::VectorDB;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::Serialize;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<RwLock<VectorDB>>,
    pub audit: Arc<RwLock<AuditChain>>,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_vectors: usize,
    pub dimension: usize,
    pub index_size: u64,
    pub hnsw_layers: usize,
    pub chain_length: usize,
    pub chain_valid: bool,
    pub latest_hash: String,
}

#[derive(Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Serialize)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: serde_json::Value,
}

#[derive(Serialize)]
pub struct AuditResponse {
    pub entries: Vec<AuditEntryJson>,
    pub total: usize,
    pub chain_valid: bool,
}

#[derive(Serialize)]
pub struct AuditEntryJson {
    pub index: u64,
    pub timestamp: String,
    pub operation: String,
    pub hash: String,
    pub prev_hash: String,
}

async fn get_stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let db = state.db.read().await;
    let audit = state.audit.read().await;
    Json(StatsResponse {
        total_vectors: db.len(),
        dimension: db.dimension(),
        index_size: db.index_size(),
        hnsw_layers: db.hnsw_layers(),
        chain_length: audit.len(),
        chain_valid: audit.verify_chain(),
        latest_hash: audit.latest_hash().to_string(),
    })
}

async fn get_audit(State(state): State<AppState>) -> Json<AuditResponse> {
    let audit = state.audit.read().await;
    let recent = audit.recent(20);
    let entries: Vec<AuditEntryJson> = recent
        .into_iter()
        .map(|b| AuditEntryJson {
            index: b.index,
            timestamp: b.timestamp,
            operation: format!("{:?}", b.operation),
            hash: b.hash,
            prev_hash: b.prev_hash,
        })
        .collect();
    let chain_valid = audit.verify_chain();
    Json(AuditResponse {
        entries,
        total: audit.len(),
        chain_valid,
    })
}

async fn dashboard() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

pub fn create_router(db: Arc<RwLock<VectorDB>>, audit: Arc<RwLock<AuditChain>>) -> Router {
    let state = AppState { db, audit };
    Router::new()
        .route("/", get(dashboard))
        .route("/api/stats", get(get_stats))
        .route("/api/audit", get(get_audit))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

static DASHBOARD_HTML: &str = r##"<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏛️ 壁画守护者 | Mural Guardian</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #faf8f5; color: #2c2c2c; }
        .header { background: linear-gradient(135deg, #8B4513 0%, #D2691E 50%, #CD853F 100%); color: white; padding: 2rem; text-align: center; }
        .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.9; font-size: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 1.5rem; }
        .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 2px solid #e0d5c8; padding-bottom: 0.5rem; }
        .tab { padding: 0.6rem 1.2rem; cursor: pointer; border-radius: 8px 8px 0 0; border: none; background: transparent; font-size: 0.95rem; color: #666; transition: all 0.2s; }
        .tab:hover { background: #f0e8dd; }
        .tab.active { background: white; color: #8B4513; font-weight: 600; box-shadow: 0 -2px 0 #8B4513; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .stat-card { background: white; border-radius: 12px; padding: 1.2rem; box-shadow: 0 2px 8px rgba(139,69,19,0.08); border-left: 4px solid #8B4513; }
        .stat-card .label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-card .value { font-size: 1.8rem; font-weight: 700; color: #2c2c2c; margin-top: 0.3rem; }
        .stat-card .value.accent { color: #8B4513; }
        .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(139,69,19,0.08); margin-bottom: 1rem; }
        .card h3 { color: #8B4513; margin-bottom: 1rem; font-size: 1.1rem; }
        .audit-entry { padding: 0.8rem; border-bottom: 1px solid #f0e8dd; font-size: 0.85rem; }
        .audit-entry:last-child { border-bottom: none; }
        .audit-entry .idx { font-weight: 600; color: #8B4513; }
        .audit-entry .op { color: #555; margin-top: 0.2rem; }
        .audit-entry .hash { font-family: monospace; font-size: 0.75rem; color: #999; margin-top: 0.2rem; }
        .badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
        .badge-green { background: #e8f5e9; color: #2e7d32; }
        .badge-red { background: #ffebee; color: #c62828; }
        .empty-state { text-align: center; padding: 2rem; color: #999; }
        .footer { text-align: center; padding: 1.5rem; color: #999; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ 壁画守护者</h1>
        <p>Mural Guardian — 联邦学习驱动的壁画智能修复平台</p>
    </div>
    <div class="container">
        <div class="tabs">
            <button class="tab active" onclick="switchTab('overview')">📊 总览</button>
            <button class="tab" onclick="switchTab('audit')">⛓️ 审计链</button>
        </div>
        <div id="tab-overview" class="tab-content active">
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card"><div class="label">壁画特征向量</div><div class="value" id="totalVectors">-</div></div>
                <div class="stat-card"><div class="label">向量维度</div><div class="value accent" id="dimension">-</div></div>
                <div class="stat-card"><div class="label">HNSW 索引</div><div class="value" id="indexSize">-</div></div>
                <div class="stat-card"><div class="label">HNSW 层级</div><div class="value" id="hnswLayers">-</div></div>
                <div class="stat-card"><div class="label">审计链长度</div><div class="value accent" id="chainLength">-</div></div>
                <div class="stat-card"><div class="label">链完整性</div><div class="value" id="chainValid">-</div></div>
            </div>
        </div>
        <div id="tab-audit" class="tab-content">
            <div class="card">
                <h3>⛓️ 修复审计日志</h3>
                <div id="auditLog"><div class="empty-state">加载中...</div></div>
            </div>
        </div>
    </div>
    <div class="footer">
        <p>🏛️ 壁画守护者 | Mural Guardian v2</p>
        <p>Powered by Rust HNSW + gRPC + SHA-256 Audit Chain</p>
    </div>
    <script>
        async function refreshStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('totalVectors').textContent = data.total_vectors;
                document.getElementById('dimension').textContent = data.dimension;
                document.getElementById('indexSize').textContent = data.index_size;
                document.getElementById('hnswLayers').textContent = data.hnsw_layers;
                document.getElementById('chainLength').textContent = data.chain_length;
                const cv = document.getElementById('chainValid');
                cv.innerHTML = data.chain_valid
                    ? '<span class="badge badge-green">✓ 有效</span>'
                    : '<span class="badge badge-red">✗ 篡改</span>';
            } catch (e) {
                console.error('Stats error:', e);
            }
        }
        async function loadAudit() {
            try {
                const res = await fetch('/api/audit');
                const data = await res.json();
                if (data.entries.length === 0) {
                    document.getElementById('auditLog').innerHTML = '<div class="empty-state">暂无审计记录</div>';
                    return;
                }
                let html = '';
                data.entries.forEach(b => {
                    html += `<div class="audit-entry">
                        <span class="idx">#${b.index}</span>
                        <span style="float:right;color:#999;font-size:0.8rem">${b.timestamp.substring(0,19)}</span>
                        <div class="op">${escapeHtml(b.operation)}</div>
                        <div class="hash">hash: ${b.hash.substring(0,16)}... → prev: ${b.prev_hash.substring(0,16)}...</div>
                    </div>`;
                });
                document.getElementById('auditLog').innerHTML = html;
            } catch (e) {
                document.getElementById('auditLog').innerHTML = '<div class="empty-state">Error loading audit</div>';
            }
        }
        function switchTab(name) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelector(`.tab[onclick="switchTab('${name}')"]`).classList.add('active');
            document.getElementById('tab-' + name).classList.add('active');
        }
        function escapeHtml(str) {
            const div = document.createElement('div');
            div.textContent = str;
            return div.innerHTML;
        }
        refreshStats();
        loadAudit();
        setInterval(refreshStats, 5000);
        setInterval(loadAudit, 10000);
    </script>
</body>
</html>"##;
