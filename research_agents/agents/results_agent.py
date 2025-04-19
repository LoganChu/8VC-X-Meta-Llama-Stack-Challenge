from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput
from llama_index import Document
import pandas as pd
from pathlib import Path

class ResultsAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        index_path: Optional[str] = "indices/results_index"
    ):
        super().__init__(model_path, index_path)
        self.section_type = "results"
        self.prompt_template = """
        Write a detailed results section for a research paper based on the following information:
        
        Research Topic: {topic}
        Data Analysis: {data_analysis}
        Key Findings: {findings}
        Methods Summary: {methods_summary}
        Additional Context: {context}
        
        The results section should:
        1. Present the data clearly and objectively
        2. Include appropriate statistical analyses
        3. Highlight significant findings
        4. Reference tables and figures appropriately
        5. Maintain consistency with the methods section
        
        Please write in a clear, academic style suitable for publication.
        """

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate results section"""
        # Query the knowledge base for relevant data analysis
        relevant_data = self.query_index(input_data.text)
        data_analysis = "\n".join([node["text"] for node in relevant_data])
        
        # Extract key findings from the knowledge base
        findings = []
        for node in relevant_data:
            if node["metadata"].get("type") == "finding":
                findings.append(node["text"])
        
        # Prepare the prompt
        prompt = self.prompt_template.format(
            topic=input_data.text,
            data_analysis=data_analysis,
            findings="\n".join(findings),
            methods_summary=input_data.metadata.get("methods_summary", ""),
            context=input_data.metadata.get("context", "")
        )
        
        # Generate response using LlamaCPP
        response = self.llm.complete(prompt)
        
        return AgentOutput(
            text=response.text,
            confidence=0.9,
            metadata={
                "section_type": self.section_type,
                "sources": [node["metadata"] for node in relevant_data]
            }
        )

    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with results-specific training data"""
        # Convert training data to documents
        documents = []
        for item in training_data:
            doc = Document(
                text=item["input"],
                metadata={
                    "output": item["output"],
                    "type": "training_data",
                    "source": "fine_tuning",
                    "data_type": item.get("data_type", ""),
                    "statistical_method": item.get("statistical_method", "")
                }
            )
            documents.append(doc)
        
        # Add documents to the index
        self.add_documents(documents)
        
        # Save the updated index
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(self.index_path)

    def process_statistical_data(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process statistical data into documents"""
        documents = []
        for item in data:
            if item["type"] == "statistical_result":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "statistical_result",
                        "source": "statistical_analysis",
                        "method": item.get("method", ""),
                        "p_value": item.get("p_value", ""),
                        "effect_size": item.get("effect_size", ""),
                        "confidence_interval": item.get("confidence_interval", "")
                    }
                )
                documents.append(doc)
            elif item["type"] == "finding":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "finding",
                        "source": "data_analysis",
                        "significance": item.get("significance", ""),
                        "implication": item.get("implication", "")
                    }
                )
                documents.append(doc)
        return documents 