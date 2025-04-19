from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput
from llama_index import Document
from llama_index.schema import TextNode
import pandas as pd
import numpy as np
from pathlib import Path

class MethodsAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        index_path: Optional[str] = "indices/methods_index"
    ):
        super().__init__(model_path, index_path)
        self.section_type = "methods"
        self.prompt_template = """
        Write a detailed methods section for a research paper based on the following information:
        
        Research Topic: {topic}
        Data Provided: {data}
        Additional Context: {context}
        Relevant Knowledge: {knowledge}
        
        The methods section should include:
        1. Study Design
        2. Data Collection
        3. Analysis Methods
        4. Statistical Methods
        5. Ethical Considerations
        
        Please write in a clear, academic style suitable for publication.
        """

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate methods section"""
        # Query the knowledge base for relevant information
        relevant_knowledge = self.query_index(input_data.text)
        knowledge_text = "\n".join([node["text"] for node in relevant_knowledge])
        
        # Prepare the prompt
        prompt = self.prompt_template.format(
            topic=input_data.text,
            data=input_data.metadata.get("data", ""),
            context=input_data.metadata.get("context", ""),
            knowledge=knowledge_text
        )
        
        # Generate response using LlamaCPP
        response = self.llm.complete(prompt)
        
        return AgentOutput(
            text=response.text,
            confidence=0.9,  # This could be calculated based on model probabilities
            metadata={
                "section_type": self.section_type,
                "sources": [node["metadata"] for node in relevant_knowledge]
            }
        )

    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with methods-specific training data"""
        # Convert training data to documents
        documents = []
        for item in training_data:
            doc = Document(
                text=item["input"],
                metadata={
                    "output": item["output"],
                    "type": "training_data",
                    "source": "fine_tuning"
                }
            )
            documents.append(doc)
        
        # Add documents to the index
        self.add_documents(documents)
        
        # Save the updated index
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(self.index_path)

    def process_multimodal_data(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process multimodal data (e.g., images, tables) relevant to methods"""
        documents = []
        for item in data:
            if item["type"] == "table":
                # Process table data
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
                # Process image data
                doc = Document(
                    text=f"Image description: {item['description']}",
                    metadata={
                        "type": "image",
                        "source": "multimodal_input",
                        "description": item.get("description", "")
                    }
                )
                documents.append(doc)
        
        return documents 