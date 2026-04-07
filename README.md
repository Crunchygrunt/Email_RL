---
title: Algorithm Architects Email RL
emoji: 📬
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 8000
tags:
  - openenv
---

Email Triage RL -- OpenEnv Environment

An OpenEnv environment where AI agents learn to prioritize, categorize, and route business emails for a fictional B2B SaaS company. This simulates a genuine daily task performed by support teams, operations staff, and account managers at every company.

Motivation

Email triage is one of the most universal knowledge-work bottlenecks -- and among the hardest to automate well. It demands multi-dimensional reasoning: urgency assessment, business context, sender relationships, security threat detection, and organizational routing. This environment tests whether LLM agents can develop the nuanced judgment needed to handle a realistic corporate inbox, including edge cases that trip up even experienced humans.

Why this matters for the RL/agent community:

- Genuine, high-frequency business task -- not a toy problem
- Multi-dimensional decision space (priority x category x route) with meaningful interactions
- Natural difficulty progression from pattern matching to strategic reasoning
- Rich partial-credit reward signal enables meaningful gradient for training (GRPO-compatible)
- Novel mechanics: phishing detection, cross-email dependencies, and escalation consequences

Environment Description

The agent manages the inbox for Nexora Technologies, a B2B SaaS project management company. Each episode presents 10 emails drawn from a balanced pool of 7 categories, including business-critical emails requiring human sign-off, phishing attempts disguised as legitimate messages, and linked incident chains.

The agent must decide three things for each email:

- Priority -- how urgently should this be handled?
- Category -- what type of email is this?
- Route -- which team or queue should receive it?

Action Space

The agent outputs three string fields per email:

priority: low, medium, high, urgent  
category: spam, newsletter, support, sales, internal, billing, security  
route: inbox, archive, support_team, sales_team, security_team, billing_team, trash, human_review  

The agent responds using XML tags:

<priority>urgent</priority>
<category>security</category>
<route>security_team</route>

Observation Space

Each observation provides:

- email_id (string): Unique email identifier
- email_subject (string): Subject line
- email_sender (string): Sender address
- email_body (string): Full body text
- last_priority_correct (bool or null)
- last_category_correct (bool or null)
- last_route_correct (bool or null)
- emails_remaining (int)
- current_streak (int)
- metadata.linked_incident (bool)

Security note: The observation does NOT contain ground truth for the current email. Ground truth for the previously graded email is embedded in metadata.graded_true_* keys after the agent acts, for client-side grader use only.

Tasks

Task 1: Spam Detection (Easy)  
Binary classification -- is this spam/phishing or legitimate?  
Score: 1.0 for correct, 0.0 for incorrect  
Success threshold: 0.6  

Task 2: Priority Classification (Medium)  
Assign the exact urgency level  
Score: 1.0 for exact match, 0.0 otherwise  
Success threshold: 0.5  

Task 3: Full Triage (Hard)  
Weighted score across all three dimensions  
Score normalized to [0.0, 1.0]  
Success threshold: 0.4  

Task 4: Critical Escalation (Hard)  
Identify business-critical emails and route to human_review  
Score: 1.0 for correct, 0.0 otherwise  
Success threshold: 0.6  

Reward Design

Base Score (per email):

- Priority correct: +1.0
- Category correct: +0.5
- Route correct: +0.3
- Format bonus: +0.1
- Perfect bonus: +0.2

Max base: 2.1

Reward Shaping:

- Urgency multiplier: x0.8 - 2.0
- Streak bonus: +0.3
- Dependency bonus: +0.4
- Overload penalty: -0.5
- Escalation multiplier: /1.5

Anti-Exploit Protections

- Ground truth is never exposed before the agent acts
- Escalation injections dynamically extend episodes
- Phishing emails bypass simple keyword matching

Novel Environment Mechanics

1. Phishing Detection  
Sophisticated phishing templates mimicking real communications

2. Cross-Email Dependencies  
Linked incidents require consistent routing

3. Escalation Consequences  
Misclassified urgent emails trigger penalty follow-ups

Setup Instructions

Local Development:

git clone <repo-url>  
cd Email_OpenEnv/Email_RL  

uv sync  
or  
pip install -e .  

uvicorn server.app:app --host 0.0.0.0 --port 8000  

Docker:

docker build -t email-triage-env .  
docker run -p 8000:8000 email-triage-env  

Run Baseline Inference:

export API_BASE_URL="https://router.huggingface.co/v1"  
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"  
export HF_TOKEN="your-token-here"  
export EMAIL_RL_SERVER_URL="http://localhost:8000"  

python inference.py  

API Endpoints

- GET /  
- GET /health  
- POST /reset  
- POST /step  
- GET /state  
- GET /schema  
- WS /ws  

Baseline Scores

- spam-detection (Easy): 1.000
- priority-classification (Medium): 0.500
- full-triage (Hard): 0.615
- critical-escalation (Hard): 0.900

Project Structure

Email_RL/  
    server/  
        __init__.py  
        app.py  
        Email_RL_environment.py  
    __init__.py  
    models.py  
    client.py  
    inference.py  
    train.py  
    openenv.yaml  
    Dockerfile  
    pyproject.toml  
    .env.example  
    README.md  

Validation

openenv validate  

curl http://localhost:8000/  
curl -X POST http://localhost:8000/reset  
curl http://localhost:8000/state  