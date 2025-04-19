from typing import Dict, List, Any, Optional
from pathlib import Path
from llama_index import Document, VectorStoreIndex, ServiceContext
from .base_agent import BaseAgent, AgentInput, AgentOutput
from .literature_agent import LiteratureAgent
from .methods_agent import MethodsAgent
from .results_agent import ResultsAgent
from .discussion_agent import DiscussionAgent
from .conclusion_agent import ConclusionAgent
import json

class CoordinatorAgent:
    def __init__(
        self,
        model_path: str = "models/llama-2-7b-chat.gguf",
        shared_knowledge_path: str = "indices/shared_knowledge",
        output_dir: str = "output"
    ):
        self.model_path = model_path
        self.shared_knowledge_path = Path(shared_knowledge_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized agents
        self.agents = {
            "literature": LiteratureAgent(model_path, "indices/literature_index"),
            "methods": MethodsAgent(model_path, "indices/methods_index"),
            "results": ResultsAgent(model_path, "indices/results_index"),
            "discussion": DiscussionAgent(model_path, "indices/discussion_index"),
            "conclusion": ConclusionAgent(model_path, "indices/conclusion_index")
        }
        
        # Initialize shared knowledge base
        self.shared_knowledge = VectorStoreIndex([])
        self._setup_shared_knowledge()

    def _setup_shared_knowledge(self):
        """Setup the shared knowledge base for all agents"""
        if self.shared_knowledge_path.exists():
            # Load existing knowledge
            self.shared_knowledge = VectorStoreIndex.load_from_disk(
                str(self.shared_knowledge_path)
            )
        else:
            # Create new knowledge base
            self.shared_knowledge_path.mkdir(parents=True, exist_ok=True)
            self.shared_knowledge.save_to_disk(str(self.shared_knowledge_path))

    def _update_shared_knowledge(self, section: str, content: str, metadata: Dict[str, Any]):
        """Update the shared knowledge base with new information"""
        doc = Document(
            text=content,
            metadata={
                "section": section,
                **metadata
            }
        )
        self.shared_knowledge.insert_nodes([doc])
        self.shared_knowledge.save_to_disk(str(self.shared_knowledge_path))

    def _get_relevant_knowledge(self, query: str, section: str) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the shared knowledge base"""
        query_engine = self.shared_knowledge.as_query_engine()
        response = query_engine.query(query)
        return [
            {
                "text": node.text,
                "metadata": node.metadata
            }
            for node in response.source_nodes
            if node.metadata.get("section") != section
        ]

    async def write_paper(
        self,
        topic: str,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, str]:
        """Coordinate the writing of a complete research paper"""
        paper_sections = {}
        previous_sections = []
        
        # Define the writing order
        sections_order = [
            "literature",
            "methods",
            "results",
            "discussion",
            "conclusion"
        ]
        
        for section in sections_order:
            agent = self.agents[section]
            
            # Prepare input with context from previous sections
            input_data = AgentInput(
                text=topic,
                metadata={
                    "data": data,
                    "context": context,
                    "previous_sections": previous_sections
                }
            )
            
            # Get relevant knowledge from other sections
            relevant_knowledge = self._get_relevant_knowledge(topic, section)
            if relevant_knowledge:
                input_data.metadata["relevant_knowledge"] = relevant_knowledge
            
            # Generate section
            output = await agent.process_input(input_data)
            
            # Update shared knowledge
            self._update_shared_knowledge(
                section=section,
                content=output.text,
                metadata=output.metadata
            )
            
            # Store section and update context
            paper_sections[section] = output.text
            previous_sections.append(output.text)
            
            # Save intermediate results
            self._save_intermediate_results(paper_sections, section)
        
        return paper_sections

    def _save_intermediate_results(self, paper_sections: Dict[str, str], current_section: str):
        """Save intermediate results to files"""
        # Save current state
        state_file = self.output_dir / "paper_state.json"
        with open(state_file, "w") as f:
            json.dump(paper_sections, f, indent=2)
        
        # Save current section
        section_file = self.output_dir / f"{current_section}_section.txt"
        with open(section_file, "w") as f:
            f.write(paper_sections[current_section])
        
        # Save complete paper
        complete_paper = "\n\n".join([
            f"=== {section.upper()} ===\n{content}"
            for section, content in paper_sections.items()
        ])
        paper_file = self.output_dir / "complete_paper.txt"
        with open(paper_file, "w") as f:
            f.write(complete_paper)

    def get_paper_progress(self) -> Dict[str, Any]:
        """Get the current progress of the paper writing"""
        state_file = self.output_dir / "paper_state.json"
        if state_file.exists():
            with open(state_file) as f:
                return json.load(f)
        return {} 