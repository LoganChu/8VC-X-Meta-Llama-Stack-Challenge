from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput
from llama_index import Document
import pandas as pd
from pathlib import Path

class ConclusionAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        index_path: Optional[str] = "indices/conclusion_index"
    ):
        super().__init__(model_path, index_path)
        self.section_type = "conclusion"
        self.prompt_template = """
        Write a comprehensive conclusion section for a research paper based on the following information:
        
        Research Topic: {topic}
        Key Findings: {findings}
        Main Discussion Points: {discussion_points}
        Implications: {implications}
        Additional Context: {context}
        
        The conclusion section should:
        1. Summarize the main findings
        2. Restate the significance of the research
        3. Highlight key implications
        4. Suggest practical applications
        5. End with a strong closing statement
        
        Please write in a clear, academic style suitable for publication.
        """

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate conclusion section"""
        # Query the knowledge base for relevant information
        relevant_info = self.query_index(input_data.text)
        
        # Extract key findings, discussion points, and implications
        findings = []
        discussion_points = []
        implications = []
        
        for node in relevant_info:
            if node["metadata"].get("type") == "finding":
                findings.append(node["text"])
            elif node["metadata"].get("type") == "discussion_point":
                discussion_points.append(node["text"])
            elif node["metadata"].get("type") == "implication":
                implications.append(node["text"])
        
        # Prepare the prompt
        prompt = self.prompt_template.format(
            topic=input_data.text,
            findings="\n".join(findings),
            discussion_points="\n".join(discussion_points),
            implications="\n".join(implications),
            context=input_data.metadata.get("context", "")
        )
        
        # Generate response using LlamaCPP
        response = self.llm.complete(prompt)
        
        return AgentOutput(
            text=response.text,
            confidence=0.9,
            metadata={
                "section_type": self.section_type,
                "sources": [node["metadata"] for node in relevant_info]
            }
        )

    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with conclusion-specific training data"""
        # Convert training data to documents
        documents = []
        for item in training_data:
            doc = Document(
                text=item["input"],
                metadata={
                    "output": item["output"],
                    "type": "training_data",
                    "source": "fine_tuning",
                    "conclusion_type": item.get("conclusion_type", ""),
                    "impact_level": item.get("impact_level", "")
                }
            )
            documents.append(doc)
        
        # Add documents to the index
        self.add_documents(documents)
        
        # Save the updated index
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(self.index_path)

    def process_conclusion_elements(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process conclusion elements into documents"""
        documents = []
        for item in data:
            if item["type"] == "summary_point":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "summary_point",
                        "source": "analysis",
                        "importance": item.get("importance", ""),
                        "connection": item.get("connection", "")
                    }
                )
                documents.append(doc)
            elif item["type"] == "application":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "application",
                        "source": "analysis",
                        "field": item.get("field", ""),
                        "practicality": item.get("practicality", "")
                    }
                )
                documents.append(doc)
        return documents 