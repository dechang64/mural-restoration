use crate::blockchain::{AuditChain, DefectType, OperationType, RestorationMethod};
use crate::db::VectorDB;
use crate::proto::mural_vector_db_server::{MuralVectorDb, MuralVectorDbServer};
use crate::proto::{
    DeleteRequest, DeleteResponse, GetAuditRequest, GetAuditResponse, AuditEntry,
    InsertRequest, InsertResponse, RestorationLogRequest, RestorationLogResponse,
    SearchRequest, SearchResponse, SearchResult, StatsRequest, StatsResponse,
};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

pub struct MuralVectorDbService {
    db: Arc<RwLock<VectorDB>>,
    audit: Arc<RwLock<AuditChain>>,
}

impl MuralVectorDbService {
    pub fn new(db: Arc<RwLock<VectorDB>>, audit: Arc<RwLock<AuditChain>>) -> Self {
        Self { db, audit }
    }
}

fn parse_defect_type(s: &str) -> DefectType {
    match s.to_lowercase().as_str() {
        "flaking" => DefectType::Flaking,
        "saline" => DefectType::Saline,
        "hollowing" => DefectType::Hollowing,
        "cracking" => DefectType::Cracking,
        "fading" => DefectType::Fading,
        "mold" => DefectType::Mold,
        _ => DefectType::Flaking,
    }
}

fn parse_restoration_method(s: &str) -> RestorationMethod {
    match s.to_lowercase().as_str() {
        "inpainting" => RestorationMethod::Inpainting,
        "color_transfer" => RestorationMethod::ColorTransfer,
        "texture_synthesis" => RestorationMethod::TextureSynthesis,
        "virtual_restoration" => RestorationMethod::VirtualRestoration,
        _ => RestorationMethod::Inpainting,
    }
}

#[tonic::async_trait]
impl MuralVectorDb for MuralVectorDbService {
    async fn insert(
        &self,
        req: Request<InsertRequest>,
    ) -> Result<Response<InsertResponse>, Status> {
        let vectors = req.into_inner().vectors;
        let count = vectors.len();
        let mut db = self.db.write().await;
        for v in vectors {
            let metadata: HashMap<String, String> = v.metadata;
            let entry = crate::db::VectorEntry {
                id: v.id,
                values: v.values,
                metadata,
            };
            if let Err(e) = db.insert(entry) {
                return Err(Status::invalid_argument(e.to_string()));
            }
        }
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Insert { count });
        Ok(Response::new(InsertResponse { inserted: count as i32 }))
    }

    async fn search(
        &self,
        req: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        let req = req.into_inner();
        let k = req.k.max(1) as usize;
        let query: Vec<f32> = req.query;
        let mut db = self.db.write().await;
        let results = db.search(&query, k);
        let search_results: Vec<SearchResult> = results
            .into_iter()
            .map(|(id, distance, metadata)| SearchResult {
                id,
                values: vec![],
                distance,
                metadata,
            })
            .collect();
        let result_count = search_results.len();
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Search { k, result_count });
        Ok(Response::new(SearchResponse { results: search_results }))
    }

    async fn delete(
        &self,
        req: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        let ids = req.into_inner().ids;
        let count = ids.len();
        let mut db = self.db.write().await;
        let deleted = db.delete(&ids) as i32;
        drop(db);
        let mut audit = self.audit.write().await;
        audit.append(OperationType::Delete { count });
        Ok(Response::new(DeleteResponse { deleted }))
    }

    async fn stats(
        &self,
        _req: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let db = self.db.read().await;
        Ok(Response::new(StatsResponse {
            total_vectors: db.len() as i64,
            dimension: db.dimension() as i32,
            index_size: db.index_size() as i64,
        }))
    }

    async fn log_restoration(
        &self,
        req: Request<RestorationLogRequest>,
    ) -> Result<Response<RestorationLogResponse>, Status> {
        let req = req.into_inner();
        let defect_type = parse_defect_type(&req.defect_type);
        let method = parse_restoration_method(&req.restoration_method);
        let block = {
            let mut audit = self.audit.write().await;
            audit.append(OperationType::Restoration {
                mural_id: req.mural_id.clone(),
                defect_type,
                method,
                reference_id: req.reference_id.clone(),
                expert_id: req.expert_id.clone(),
                confidence: req.confidence,
            });
            audit.recent(1)[0].clone()
        };
        Ok(Response::new(RestorationLogResponse {
            success: true,
            block_hash: block.hash,
            block_index: block.index as i64,
        }))
    }

    async fn get_audit(
        &self,
        req: Request<GetAuditRequest>,
    ) -> Result<Response<GetAuditResponse>, Status> {
        let limit = req.into_inner().limit.max(1) as usize;
        let audit = self.audit.read().await;
        let recent = audit.recent(limit);
        let entries: Vec<AuditEntry> = recent
            .into_iter()
            .map(|b| AuditEntry {
                index: b.index as i64,
                timestamp: b.timestamp.clone(),
                operation: format!("{:?}", b.operation),
                hash: b.hash.clone(),
                prev_hash: b.prev_hash.clone(),
            })
            .collect();
        let chain_valid = audit.verify_chain();
        let total = audit.len() as i64;
        Ok(Response::new(GetAuditResponse {
            entries,
            total,
            chain_valid,
        }))
    }
}

pub fn create_server(
    db: Arc<RwLock<VectorDB>>,
    audit: Arc<RwLock<AuditChain>>,
) -> MuralVectorDbServer<MuralVectorDbService> {
    MuralVectorDbServer::new(MuralVectorDbService::new(db, audit))
}
