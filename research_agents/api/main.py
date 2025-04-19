from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import json
import os
from ..agents.methods_agent import MethodsAgent
from ..agents.results_agent import ResultsAgent
from ..agents.discussion_agent import DiscussionAgent

app = FastAPI(title="Research Paper Writing Multi-Agent System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
methods_agent = MethodsAgent()
results_agent = ResultsAgent()
discussion_agent = DiscussionAgent()

class ResearchInput(BaseModel):
    topic: str
    data: Optional[dict] = None
    context: Optional[str] = None
    multimodal_data: Optional[List[dict]] = None

class ResearchOutput(BaseModel):
    methods: str
    results: str
    discussion: str
    confidence_scores: dict

@app.post("/generate_paper", response_model=ResearchOutput)
async def generate_paper(input_data: ResearchInput):
    """Generate a complete research paper using all agents"""
    # Process with methods agent
    methods_input = {
        "text": input_data.topic,
        "metadata": {
            "data": input_data.data,
            "context": input_data.context
        },
        "multimodal_data": input_data.multimodal_data
    }
    methods_output = await methods_agent.process_input(methods_input)
    
    # Process with results agent
    results_input = {
        "text": methods_output.text,
        "metadata": {
            "data": input_data.data,
            "context": input_data.context
        },
        "multimodal_data": input_data.multimodal_data
    }
    results_output = await results_agent.process_input(results_input)
    
    # Process with discussion agent
    discussion_input = {
        "text": f"{methods_output.text}\n{results_output.text}",
        "metadata": {
            "data": input_data.data,
            "context": input_data.context
        },
        "multimodal_data": input_data.multimodal_data
    }
    discussion_output = await discussion_agent.process_input(discussion_input)
    
    return ResearchOutput(
        methods=methods_output.text,
        results=results_output.text,
        discussion=discussion_output.text,
        confidence_scores={
            "methods": methods_output.confidence,
            "results": results_output.confidence,
            "discussion": discussion_output.confidence
        }
    )

@app.post("/fine_tune/{agent_type}")
async def fine_tune_agent(
    agent_type: str,
    training_data: UploadFile = File(...),
    epochs: int = Form(3),
    batch_size: int = Form(4)
):
    """Fine-tune a specific agent with provided training data"""
    # Read and parse training data
    content = await training_data.read()
    training_data_json = json.loads(content)
    
    # Select appropriate agent
    agent = {
        "methods": methods_agent,
        "results": results_agent,
        "discussion": discussion_agent
    }.get(agent_type)
    
    if not agent:
        return {"error": f"Invalid agent type: {agent_type}"}
    
    # Fine-tune the agent
    await agent.fine_tune(
        training_data=training_data_json,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return {"status": "success", "message": f"{agent_type} agent fine-tuned successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"} 