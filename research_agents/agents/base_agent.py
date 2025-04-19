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
    metadata: Dict[str, Any] = {}
    multimodal_data: Optional[List[Dict[str, Any]]] = None
    previous_sections: Optional[List[str]] = None

class AgentOutput(BaseModel):
    """Base output model for agent communication"""
    text: str
    confidence: float
    metadata: Dict[str, Any] = {}
    citations: Optional[List[str]] = None

class BaseAgent(ABC):
    def __init__(
        self,
        model_path: str,
        index_path: str,
        system_prompt: str = "",
        section_type: str = "general"
    ):
        self.model_path = model_path
        self.index_path = Path(index_path)
        self.system_prompt = system_prompt
        self.section_type = section_type
        
        # Initialize LLM
        self.llm = LlamaCPP(
            model_path=model_path,
            temperature=0.7,
            max_new_tokens=2048,
            context_window=4096,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": 1}
        )
        
        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Initialize service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        
        # Initialize or load knowledge base
        self._setup_knowledge_base()

    def _setup_knowledge_base(self):
        """Setup the agent's knowledge base"""
        if self.index_path.exists():
            self.knowledge_base = VectorStoreIndex.load_from_disk(
                str(self.index_path),
                service_context=self.service_context
            )
        else:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.knowledge_base = VectorStoreIndex([], service_context=self.service_context)
            self.knowledge_base.save_to_disk(str(self.index_path))

    def _get_section_prompt(self, input_data: AgentInput) -> str:
        """Get the prompt for the specific section"""
        # Base prompt with system instructions
        prompt = f"{self.system_prompt}\n\n"
        
        # Add context from previous sections if available
        if "previous_sections" in input_data.metadata:
            prompt += "Previous sections of the paper:\n"
            for section in input_data.metadata["previous_sections"]:
                prompt += f"{section}\n\n"
        
        # Add relevant knowledge from other sections if available
        if "relevant_knowledge" in input_data.metadata:
            prompt += "Relevant information from other sections:\n"
            for knowledge in input_data.metadata["relevant_knowledge"]:
                prompt += f"{knowledge['text']}\n\n"
        
        # Add the current task
        prompt += f"Write the {self.section_type} section for the research paper on: {input_data.text}\n"
        
        # Add any additional context
        if "context" in input_data.metadata:
            prompt += f"Additional context: {input_data.metadata['context']}\n"
        
        return prompt

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate output"""
        # Get the section-specific prompt
        prompt = self._get_section_prompt(input_data)
        
        # Generate response
        response = self.llm.complete(prompt)
        
        # Create output
        output = AgentOutput(
            text=response.text,
            metadata={
                "section_type": self.section_type,
                "prompt": prompt,
                "input_metadata": input_data.metadata
            }
        )
        
        # Update knowledge base
        self._update_knowledge_base(output)
        
        return output

    def _update_knowledge_base(self, output: AgentOutput):
        """Update the agent's knowledge base with new information"""
        # Add the generated content to the knowledge base
        self.knowledge_base.insert_nodes([output.text])
        self.knowledge_base.save_to_disk(str(self.index_path))

    def get_knowledge(self, query: str) -> str:
        """Retrieve relevant knowledge from the agent's knowledge base"""
        query_engine = self.knowledge_base.as_query_engine()
        response = query_engine.query(query)
        return response.response

    def add_documents(self, documents: List[Document]):
        """Add documents to the agent's knowledge base"""
        self.knowledge_base.insert_nodes(documents)

    def save_index(self, path: str):
        """Save the current index state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_base.storage_context.persist(persist_dir=path)

    def query_index(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query the agent's knowledge base"""
        query_engine = self.knowledge_base.as_query_engine()
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
    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with specific training data"""
        pass

    async def communicate(self, other_agent: 'BaseAgent', message: AgentInput) -> AgentOutput:
        """Communicate with another agent"""
        return await other_agent.process_input(message) 