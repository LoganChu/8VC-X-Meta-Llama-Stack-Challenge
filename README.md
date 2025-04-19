# Multi-Agent Research Paper Writing System

This system implements a collaborative multi-agent approach to research paper writing, where different agents specialize in different sections of a research paper. The system consists of three main agents:

1. Methods Agent - Specializes in writing the methods section
2. Results Agent - Specializes in writing the results section
3. Discussion Agent - Specializes in writing the discussion and conclusion sections

## Features

- Individual agents trained on specific paper sections
- Collaborative writing through agent communication
- Multi-modal input support (text, images, data)
- Fine-tuning capabilities with researcher's data
- Interactive prompting system

## Project Structure

```
research_agents/
├── agents/
│   ├── base_agent.py
│   ├── methods_agent.py
│   ├── results_agent.py
│   └── discussion_agent.py
├── models/
│   └── model_manager.py
├── utils/
│   ├── data_processor.py
│   └── prompt_templates.py
├── api/
│   └── main.py
└── config/
    └── settings.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

3. Run the API server:
```bash
uvicorn api.main:app --reload
```

## Usage

The system can be used through the API endpoints or directly through Python scripts. Each agent can be fine-tuned with specific research data and can communicate with other agents to produce a cohesive research paper.