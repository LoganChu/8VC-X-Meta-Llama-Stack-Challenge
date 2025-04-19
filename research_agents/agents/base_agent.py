from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import torch
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import Document
from pathlib import Path
import pandas as pd

class AgentInput(BaseModel):
    """Base input model for agent communication"""
    text: str
    metadata: Optional[Dict[str, Any]] = None
    multimodal_data: Optional[List[Dict[str, Any]]] = None
    previous_sections: Optional[List[str]] = None

class AgentOutput(BaseModel):
    """Base output model for agent communication"""
    text: str
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    citations: Optional[List[str]] = None

class BaseAgent(ABC):
    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_path = model_path
        self.index_path = index_path
        self.llm = None
        self.index = None
        self.section_type = None
        self._setup_llm()
        self._setup_index()

    def _setup_llm(self):
        """Setup the LlamaCPP model"""
        self.llm = LlamaCPP(
            model_path=self.model_path,
            temperature=0.7,
            max_new_tokens=1000,
            context_window=4096,
            model_kwargs={"n_gpu_layers": -1} if self.device == "cuda" else {}
        )

    def _setup_index(self):
        """Setup or load the vector index"""
        if self.index_path and Path(self.index_path).exists():
            storage_context = StorageContext.from_defaults(persist_dir=self.index_path)
            self.index = load_index_from_storage(storage_context)
        else:
            # Create a new index
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            node_parser = SimpleNodeParser.from_defaults()
            service_context = ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=embed_model,
                node_parser=node_parser
            )
            self.index = VectorStoreIndex([], service_context=service_context)

    def add_documents(self, documents: List[Document]):
        """Add documents to the agent's knowledge base"""
        self.index.insert_nodes(documents)

    def save_index(self, path: str):
        """Save the current index state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=path)

    def query_index(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the agent's knowledge base"""
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return [
            {
                "text": node.text,
                "score": node.score,
                "metadata": node.metadata
            }
            for node in response.source_nodes[:top_k]
        ]

    def process_multimodal_data(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process multimodal data into documents"""
        documents = []
        for item in data:
            if item["type"] == "table":
                df = pd.DataFrame(item["data"])
                table_text = f"Table data with {len(df)} rows and {len(df.columns)} columns:\n{df.to_string()}"
                doc = Document(
                    text=table_text,
                    metadata={
                        "type": "table",
                        "source": "multimodal_input",
                        "description": item.get("description", "")
                    }
                )
                documents.append(doc)
            elif item["type"] == "image":
                doc = Document(
                    text=f"Image description: {item['description']}",
                    metadata={
                        "type": "image",
                        "source": "multimodal_input",
                        "description": item.get("description", "")
                    }
                )
                documents.append(doc)
            elif item["type"] == "citation":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "citation",
                        "source": "citation_input",
                        "authors": item.get("authors", ""),
                        "year": item.get("year", ""),
                        "title": item.get("title", "")
                    }
                )
                documents.append(doc)
        return documents

    @abstractmethod
    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input data and generate output"""
        pass

    @abstractmethod
    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with specific training data"""
        pass

    async def communicate(self, other_agent: 'BaseAgent', message: AgentInput) -> AgentOutput:
        """Communicate with another agent"""
        return await other_agent.process_input(message) 