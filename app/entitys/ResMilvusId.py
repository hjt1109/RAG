# ResMilvusId.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any # Added Dict, Any for schema and statistics

class MilVusInfo(BaseModel):
    collection_id: str = Field(..., description="Milvus Collection ID")
    collection_name: str = Field(..., description="Milvus Collection Name")
    description: str = Field(..., description="Milvus Collection Description")
    collection_schema: List[Dict[str, Any]] = Field(..., description="Milvus Collection Schema") # Changed to List[Dict[str, Any]]
    statistics: Dict[str, Any] = Field(..., description="Milvus Collection Statistics") # Changed to Dict[str, Any]
    all_collections: List[str] = Field(..., description="All Milvus Collections") # Changed to List[str]
    total_collections: int = Field(..., description="Total Milvus Collections")
    is_exist: bool = Field(..., description="Milvus Collection Existence")