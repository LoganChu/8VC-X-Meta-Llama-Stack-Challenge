from typing import List, Dict, Any
from pathlib import Path
from ..utils.fine_tuning import FineTuningManager
from ..agents.literature_agent import LiteratureAgent

def create_sample_training_data() -> List[Dict[str, Any]]:
    """Create sample training data for fine-tuning"""
    return [
        {
            "input": "Analyze the impact of machine learning on healthcare",
            "output": "Machine learning has significantly transformed healthcare through improved diagnostics, personalized treatment plans, and predictive analytics. Recent studies have shown that ML algorithms can achieve diagnostic accuracy comparable to human experts in various medical imaging tasks.",
            "citation": "Smith et al. (2023)"
        },
        {
            "input": "Review the current state of quantum computing",
            "output": "Quantum computing represents a paradigm shift in computational capabilities. Current research focuses on overcoming decoherence challenges and developing error-correcting codes. Recent breakthroughs in superconducting qubits have shown promising results for achieving quantum supremacy.",
            "citation": "Johnson et al. (2022)"
        },
        {
            "input": "Discuss the role of blockchain in supply chain management",
            "output": "Blockchain technology has emerged as a transformative solution for supply chain transparency and traceability. Its decentralized nature enables secure, immutable record-keeping while reducing the need for intermediaries. Recent implementations have demonstrated significant improvements in efficiency and trust.",
            "citation": "Brown et al. (2023)"
        }
    ]

def main():
    # Initialize the fine-tuning manager
    base_model_path = "models/llama-2-7b-chat.gguf"
    fine_tuning_manager = FineTuningManager(
        base_model_path=base_model_path,
        output_dir="finetuned_models"
    )
    
    # Create sample training data
    training_data = create_sample_training_data()
    
    # Fine-tune the sentence transformer model
    print("Fine-tuning sentence transformer model...")
    sentence_transformer_path = fine_tuning_manager.fine_tune_sentence_transformer(
        training_data=training_data,
        epochs=3,
        batch_size=32,
        learning_rate=1e-5
    )
    print(f"Fine-tuned sentence transformer saved to: {sentence_transformer_path}")
    
    # Fine-tune the embedding adapter
    print("\nFine-tuning embedding adapter...")
    embedding_adapter_path = fine_tuning_manager.fine_tune_embedding_adapter(
        training_data=training_data,
        epochs=3,
        batch_size=32,
        learning_rate=1e-5
    )
    print(f"Fine-tuned embedding adapter saved to: {embedding_adapter_path}")
    
    # Create a service context with the fine-tuned models
    service_context = fine_tuning_manager.create_finetuned_service_context(
        finetuned_model_path=sentence_transformer_path,
        model_type="sentence_transformer"
    )
    
    # Initialize an agent with the fine-tuned model
    literature_agent = LiteratureAgent(
        model_path=base_model_path,
        index_path="indices/literature_index"
    )
    
    # Test the fine-tuned agent
    test_input = {
        "text": "Review the applications of artificial intelligence in education",
        "metadata": {
            "context": "Focus on recent developments and practical implementations"
        }
    }
    
    print("\nTesting the fine-tuned agent...")
    response = literature_agent.process_input(test_input)
    print("\nGenerated response:")
    print(response.text)
    print("\nCitations:")
    print(response.citations)

if __name__ == "__main__":
    main() 