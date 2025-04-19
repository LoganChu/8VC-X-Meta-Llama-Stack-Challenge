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
        system_prompt = """You are an expert research paper writer specializing in literature reviews.
        Your task is to write comprehensive and well-structured literature review sections that:
        1. Provide a thorough overview of existing research
        2. Identify key themes and trends
        3. Highlight gaps in current knowledge
        4. Connect previous work to the current study
        5. Use proper academic language and citations
        
        Focus on synthesizing information rather than just summarizing individual papers.
        Ensure the literature review flows logically and builds a strong foundation for the research."""
        
        super().__init__(
            model_path=model_path,
            index_path=index_path,
            system_prompt=system_prompt,
            section_type="literature_review"
        )

    async def process_input(self, input_data: AgentInput) -> AgentOutput:
        # Get relevant knowledge from the knowledge base
        relevant_knowledge = self.get_knowledge(
            f"What are the key themes and findings in the literature about {input_data.text}?"
        )
        
        # Update input metadata with relevant knowledge
        input_data.metadata["relevant_knowledge"] = [
            {"text": relevant_knowledge, "source": "knowledge_base"}
        ]
        
        # Process the input using the base agent's method
        output = await super().process_input(input_data)
        
        # Add literature-specific metadata
        output.metadata.update({
            "section_type": "literature_review",
            "key_themes": self._extract_key_themes(output.text),
            "citation_count": self._count_citations(output.text)
        })
        
        return output

    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from the literature review"""
        query = "What are the main themes discussed in this literature review?"
        response = self.llm.complete(f"Extract key themes from: {text}\n{query}")
        return [theme.strip() for theme in response.text.split("\n") if theme.strip()]

    def _count_citations(self, text: str) -> int:
        """Count the number of citations in the text"""
        # Simple citation counting - can be enhanced with more sophisticated methods
        return text.count("(") + text.count("[")

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