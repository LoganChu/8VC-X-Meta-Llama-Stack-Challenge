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
        system_prompt = """You are an expert research paper writer specializing in methods sections.
        Your task is to write clear and detailed methods sections that:
        1. Describe the research design and methodology
        2. Explain data collection procedures
        3. Detail analysis techniques
        4. Specify equipment and materials used
        5. Include ethical considerations
        
        Focus on providing enough detail for reproducibility while maintaining clarity.
        Use precise technical language and follow standard academic conventions."""
        
        super().__init__(
            model_path=model_path,
            index_path=index_path,
            system_prompt=system_prompt,
            section_type="methods"
        )

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        # Get relevant knowledge from the knowledge base
        relevant_knowledge = self.get_knowledge(
            f"What are the standard methods and protocols for {input_data.text}?"
        )
        
        # Update input metadata with relevant knowledge
        input_data.metadata["relevant_knowledge"] = [
            {"text": relevant_knowledge, "source": "knowledge_base"}
        ]
        
        # Process the input using the base agent's method
        output = await super().process_input(input_data)
        
        # Add methods-specific metadata
        output.metadata.update({
            "section_type": "methods",
            "method_categories": self._extract_method_categories(output.text),
            "technical_terms": self._extract_technical_terms(output.text)
        })
        
        return output

    def _extract_method_categories(self, text: str) -> List[str]:
        """Extract main method categories from the text"""
        query = "What are the main method categories described in this section?"
        response = self.llm.complete(f"Extract method categories from: {text}\n{query}")
        return [category.strip() for category in response.text.split("\n") if category.strip()]

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms used in the methods section"""
        query = "What are the key technical terms used in this methods section?"
        response = self.llm.complete(f"Extract technical terms from: {text}\n{query}")
        return [term.strip() for term in response.text.split("\n") if term.strip()]

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