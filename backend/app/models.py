from datetime import datetime
from pydantic import BaseModel, Field
class Chunk(BaseModel):
    chunk_id: str; contract_id: str; page_start: int; page_end: int
    section: str = ""; clause: str = ""; heading: str = ""; text: str
    token_estimate: int = 0
class Evidence(BaseModel):
    page: int; clause: str = ""; quote: str; chunk_id: str = ""
class Finding(BaseModel):
    finding_id: str; category: str; severity: str; claim: str
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning: str = ""; recommendation: str = ""; confidence: float = 0.0
    requires_human_review: bool = False; verification_status: str = "pending"
class ReviewRequest(BaseModel):
    question: str = "Review this contract for material legal and commercial risks."
class ChatRequest(BaseModel):
    message: str
class WorkspaceChatRequest(BaseModel):
    message: str
    contract_ids: list[str] = Field(default_factory=list)
class ChatMessage(BaseModel):
    role: str
    content: str
    created_at: datetime
    citations: list[Evidence] = Field(default_factory=list)
class DecisionRequest(BaseModel):
    decision: str; note: str = ""
class ContractMeta(BaseModel):
    contract_id: str; original_filename: str; sha256: str; upload_time: datetime
    workspace_id: str = "default"
    page_count: int = 0; processing_status: str; chunk_count: int = 0
    embedding_status: str = "pending"; analysis_status: str = "not_started"
