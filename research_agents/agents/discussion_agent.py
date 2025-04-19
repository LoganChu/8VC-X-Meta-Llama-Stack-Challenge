from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput
from llama_index import Document
import pandas as pd
from pathlib import Path

class DiscussionAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        index_path: Optional[str] = "indices/discussion_index"
    ):
        super().__init__(model_path, index_path)
        self.section_type = "discussion"
        self.prompt_template = """
        Write a comprehensive discussion section for a research paper based on the following information:
        
        Research Topic: {topic}
        Key Findings: {findings}
        Previous Research: {previous_research}
        Results Summary: {results_summary}
        Additional Context: {context}
        
        The discussion section should:
        1. Interpret the results in the context of the research question
        2. Compare findings with previous research
        3. Discuss implications and significance
        4. Address limitations of the study
        5. Suggest directions for future research
        
        Please write in a clear, academic style suitable for publication.
        """

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate discussion section"""
        # Query the knowledge base for relevant information
        relevant_info = self.query_index(input_data.text)
        
        # Extract key findings and previous research
        findings = []
        previous_research = []
        for node in relevant_info:
            if node["metadata"].get("type") == "finding":
                findings.append(node["text"])
            elif node["metadata"].get("type") == "citation":
                previous_research.append(node["text"])
        
        # Prepare the prompt
        prompt = self.prompt_template.format(
            topic=input_data.text,
            findings="\n".join(findings),
            previous_research="\n".join(previous_research),
            results_summary=input_data.metadata.get("results_summary", ""),
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
        """Fine-tune the agent with discussion-specific training data"""
        # Convert training data to documents
        documents = []
        for item in training_data:
            doc = Document(
                text=item["input"],
                metadata={
                    "output": item["output"],
                    "type": "training_data",
                    "source": "fine_tuning",
                    "discussion_type": item.get("discussion_type", ""),
                    "implication_level": item.get("implication_level", "")
                }
            )
            documents.append(doc)
        
        # Add documents to the index
        self.add_documents(documents)
        
        # Save the updated index
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(self.index_path)

    def process_implications(self, data: List[Dict[str, Any]]) -> List[Document]:
        """Process implications and limitations into documents"""
        documents = []
        for item in data:
            if item["type"] == "implication":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "implication",
                        "source": "analysis",
                        "level": item.get("level", ""),
                        "scope": item.get("scope", ""),
                        "impact": item.get("impact", "")
                    }
                )
                documents.append(doc)
            elif item["type"] == "limitation":
                doc = Document(
                    text=item["text"],
                    metadata={
                        "type": "limitation",
                        "source": "analysis",
                        "category": item.get("category", ""),
                        "severity": item.get("severity", ""),
                        "mitigation": item.get("mitigation", "")
                    }
                )
                documents.append(doc)
        return documents 