Email Triage RL -- OpenEnv Environment

An OpenEnv environment where AI agents learn to prioritize, categorize, and route business emails for a fictional B2B SaaS company. This simulates a genuine daily task performed by support teams, operations staff, and account managers at every company.

---

Motivation

Email triage is one of the most universal knowledge-work bottlenecks -- and among the hardest to automate well. It demands multi-dimensional reasoning: urgency assessment, business context, sender relationships, security threat detection, and organizational routing.

This environment tests whether LLM agents can develop the nuanced judgment needed to handle a realistic corporate inbox, including edge cases that trip up even experienced humans.

Why this matters for the RL/agent community:

- Genuine, high-frequency business task -- not a toy problem
- Multi-dimensional decision space (priority x category x route)
- Natural difficulty progression from pattern matching to reasoning
- Rich partial-credit reward signal (GRPO compatible)
- Includes phishing detection, dependencies, escalation effects

---

Environment Description

The agent manages the inbox for Nexora Technologies, a B2B SaaS project management company.

Each episode presents 10 emails drawn from 7 categories, including:
- business-critical emails
- phishing attempts
- linked incident chains

The agent must decide:

- Priority -- urgency level
- Category -- type of email
- Route -- destination team or queue

---

Action Space

The agent outputs three string fields:

Field: priority
Values: low, medium, high, urgent

Field: category
Values: spam, newsletter, support, sales, internal, billing, security

Field: route
Values: inbox, archive, support_team, sales_team, security_team, billing_team, trash, human_review

Example output:

<priority>urgent</priority>
<category>security</category>
<route>security_team</route>

---

Observation Space

Each observation provides:

email_id: string - Unique ID
email_subject: string - Subject
email_sender: string - Sender
email_body: string - Full content
last_priority_correct: bool or null - Previous priority correctness
last_category_correct: bool or null - Previous category correctness
last_route_correct: bool or null - Previous route correctness
emails_remaining: int - Remaining emails
current_streak: int - Consecutive perfect decisions
metadata.linked_incident: bool - Related email hint

Security note:

- Ground truth is NOT exposed before the agent acts
- Ground truth is available only after action in metadata.graded_true_* fields

---

Tasks

Task 1: Spam Detection (Easy)
- Binary classification (spam vs legitimate)
- Score: 1.0 or 0.0
- Threshold: 0.6

Task 2: Priority Classification (Medium)
- Exact urgency classification
- Score: 1.0 or 0.0
- Threshold: 0.5

Task 3: Full Triage (Hard)
- Weighted score across all fields
- Includes phishing, dependencies, escalation
- Score normalized to [0.0, 1.0]
- Threshold: 0.4

Task 4: Critical Escalation (Hard)
- Detect business-critical emails
- Route to human_review
- Penalizes over and under escalation
- Threshold: 0.6

---

Reward Design

Base Score (per email):

Priority correct: +1.0
Category correct: +0.5
Route correct: +0.3
Format bonus: +0.1
Perfect bonus: +0.2
Max base: 2.1

Reward Shaping:

Urgency multiplier: x0.8-2.0 (based on priority)
Streak bonus: +0.3 (3+ correct)
Dependency bonus: +0.4 (linked emails consistent)
Overload penalty: -0.5 (misclassify urgent/high)
Escalation multiplier: /1.5 (follow-ups)

---

Anti-Exploit Protections

- Ground truth hidden before action
- Dynamic escalation injections
- Phishing avoids keyword shortcuts

---

Environment Mechanics

1. Phishing Detection
Includes advanced phishing cases:
- CEO fraud
- Fake IT resets
- Spoofed invoices
- Fake document links

Agent must detect subtle signals, not keywords.

2. Cross-Email Dependencies
- Related emails share context
- metadata.linked_incident indicates relation
- Consistent routing earns bonus

3. Escalation Consequences
- Misclassified urgent emails trigger follow-ups
- Follow-ups carry higher penalties

---

Setup Instructions

Local Development

git clone <repo-url>
cd Email_OpenEnv/Email_RL

# Install (recommended)
uv sync

# Or pip
pip install -e .

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

Docker

docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env

---

Run Baseline Inference

export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
export EMAIL_RL_SERVER_URL="http://localhost:8000"

python inference.py

---

API Endpoints

/        GET    Health check
/health  GET    Health check
/reset   POST   Start episode
/step    POST   Submit action
/state   GET    Episode state
/schema  GET    JSON schemas
/ws      WS     WebSocket

---

Baseline Scores

spam-detection           Easy    ~0.85
priority-classification  Medium  ~0.60
full-triage              Hard    ~0.50
critical-escalation      Hard    ~0.65

---

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

---

Validation

openenv validate

curl http://localhost:8000/
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/state