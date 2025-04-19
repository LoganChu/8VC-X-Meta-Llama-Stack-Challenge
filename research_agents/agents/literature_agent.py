from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentInput, AgentOutput
from llama_index import Document
import pandas as pd
from pathlib import Path

class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        index_path: Optional[str] = "indices/literature_index"
    ):
        super().__init__(model_path, index_path)
        self.section_type = "literature_review"
        self.prompt_template = """
        Write a comprehensive literature review section for a research paper based on the following information:
        
        Research Topic: {topic}
        Previous Research: {previous_research}
        Citations: {citations}
        Additional Context: {context}
        
        The literature review should:
        1. Provide a historical context of the research area
        2. Discuss key theories and concepts
        3. Review relevant previous studies
        4. Identify research gaps
        5. Justify the current study
        
        Please write in a clear, academic style suitable for publication, and properly cite all sources.
        """

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        """Process input and generate literature review section"""
        # Query the knowledge base for relevant research
        relevant_research = self.query_index(input_data.text)
        previous_research = "\n".join([node["text"] for node in relevant_research])
        
        # Extract citations from the knowledge base
        citations = []
        for node in relevant_research:
            if node["metadata"].get("type") == "citation":
                citations.append(
                    f"{node['metadata'].get('authors', '')} ({node['metadata'].get('year', '')})"
                )
        
        # Prepare the prompt
        prompt = self.prompt_template.format(
            topic=input_data.text,
            previous_research=previous_research,
            citations=", ".join(citations),
            context=input_data.metadata.get("context", "")
        )
        
        # Generate response using LlamaCPP
        response = self.llm.complete(prompt)
        
        return AgentOutput(
            text=response.text,
            confidence=0.9,
            metadata={
                "section_type": self.section_type,
                "sources": [node["metadata"] for node in relevant_research]
            },
            citations=citations
        )

    async def fine_tune(self, training_data: List[Dict[str, Any]], **kwargs):
        """Fine-tune the agent with literature-specific training data"""
        # Convert training data to documents
        documents = []
        for item in training_data:
            doc = Document(
                text=item["input"],
                metadata={
                    "output": item["output"],
                    "type": "training_data",
                    "source": "fine_tuning",
                    "citation": item.get("citation", "")
                }
            )
            documents.append(doc)
        
        # Add documents to the index
        self.add_documents(documents)
        
        # Save the updated index
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_index(self.index_path)

    def process_citations(self, citations: List[Dict[str, Any]]) -> List[Document]:
        """Process citation data into documents"""
        documents = []
        for citation in citations:
            doc = Document(
                text=citation["text"],
                metadata={
                    "type": "citation",
                    "source": "citation_input",
                    "authors": citation.get("authors", ""),
                    "year": citation.get("year", ""),
                    "title": citation.get("title", ""),
                    "journal": citation.get("journal", ""),
                    "doi": citation.get("doi", "")
                }
            )
            documents.append(doc)
        return documents 