from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import torch
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
)
from llama_index.llms import LlamaCPP
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.node_parser import SimpleNodeParser
from llama_index.finetuning import (
    SentenceTransformersFinetuneEngine,
    EmbeddingAdapterFinetuneEngine,
)

class FineTuningManager:
    def __init__(
        self,
        base_model_path: str,
        output_dir: str = "finetuned_models",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(
        self,
        training_data: List[Dict[str, Any]],
        data_type: str = "sentence_pairs"
    ) -> List[Dict[str, Any]]:
        """Prepare training data in the required format for fine-tuning"""
        formatted_data = []
        
        if data_type == "sentence_pairs":
            for item in training_data:
                formatted_data.append({
                    "text": item["input"],
                    "label": item["output"]
                })
        elif data_type == "triplets":
            for item in training_data:
                formatted_data.append({
                    "anchor": item["input"],
                    "positive": item["output"],
                    "negative": item.get("negative", "")
                })
        
        return formatted_data

    def save_training_data(
        self,
        data: List[Dict[str, Any]],
        filename: str = "training_data.json"
    ):
        """Save training data to a file"""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def fine_tune_sentence_transformer(
        self,
        training_data: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-small-en-v1.5",
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-5
    ):
        """Fine-tune a sentence transformer model"""
        # Prepare the training data
        formatted_data = self.prepare_training_data(training_data, "sentence_pairs")
        self.save_training_data(formatted_data, "sentence_transformer_training.json")
        
        # Initialize the fine-tuning engine
        finetune_engine = SentenceTransformersFinetuneEngine(
            model_name=model_name,
            train_data=formatted_data,
            output_dir=str(self.output_dir / "sentence_transformer"),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Run the fine-tuning
        finetune_engine.finetune()
        
        return str(self.output_dir / "sentence_transformer")

    def fine_tune_embedding_adapter(
        self,
        training_data: List[Dict[str, Any]],
        base_embedding_model: str = "BAAI/bge-small-en-v1.5",
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-5
    ):
        """Fine-tune an embedding adapter"""
        # Prepare the training data
        formatted_data = self.prepare_training_data(training_data, "triplets")
        self.save_training_data(formatted_data, "embedding_adapter_training.json")
        
        # Initialize the fine-tuning engine
        finetune_engine = EmbeddingAdapterFinetuneEngine(
            base_embedding_model=base_embedding_model,
            train_data=formatted_data,
            output_dir=str(self.output_dir / "embedding_adapter"),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Run the fine-tuning
        finetune_engine.finetune()
        
        return str(self.output_dir / "embedding_adapter")

    def create_finetuned_service_context(
        self,
        finetuned_model_path: str,
        model_type: str = "sentence_transformer"
    ) -> ServiceContext:
        """Create a service context with the fine-tuned model"""
        if model_type == "sentence_transformer":
            embed_model = HuggingFaceEmbedding(model_name=finetuned_model_path)
        else:
            embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                adapter_path=finetuned_model_path
            )
        
        llm = LlamaCPP(
            model_path=self.base_model_path,
            temperature=0.7,
            max_new_tokens=1000,
            context_window=4096,
            model_kwargs={"n_gpu_layers": -1} if self.device == "cuda" else {}
        )
        
        node_parser = SimpleNodeParser.from_defaults()
        
        return ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            node_parser=node_parser
        ) 