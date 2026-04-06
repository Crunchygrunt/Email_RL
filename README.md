# Email Triage RL -- OpenEnv Environment

An OpenEnv environment where AI agents learn to **prioritize, categorize, and route business emails** for a fictional B2B SaaS company. This simulates a genuine daily task performed by support teams, operations staff, and account managers at every company.

## Motivation

Email triage is one of the most universal knowledge-work bottlenecks -- and among the hardest to automate well. It demands multi-dimensional reasoning: urgency assessment, business context, sender relationships, security threat detection, and organizational routing. This environment tests whether LLM agents can develop the nuanced judgment needed to handle a realistic corporate inbox, including edge cases that trip up even experienced humans.

**Why this matters for the RL/agent community:**

- Genuine, high-frequency business task -- not a toy problem
- Multi-dimensional decision space (priority x category x route) with meaningful interactions
- Natural difficulty progression from pattern matching to strategic reasoning
- Rich partial-credit reward signal enables meaningful gradient for training (GRPO-compatible)
- Novel mechanics: phishing detection, cross-email dependencies, and escalation consequences

---

## Environment Description

The agent manages the inbox for Nexora Technologies, a B2B SaaS project management company. Each episode presents 10 emails drawn from a balanced pool of 7 categories, including business-critical emails requiring human sign-off, phishing attempts disguised as legitimate messages, and linked incident chains. The agent must decide three things for each email:

1. **Priority** -- how urgently should this be handled?
2. **Category** -- what type of email is this?
3. **Route** -- which team or queue should receive it?

---

## Action Space

The agent outputs three string fields per email:

| Field      | Values                                                                                    |
|------------|-------------------------------------------------------------------------------------------|
| `priority` | `low`, `medium`, `high`, `urgent`                                                         |
| `category` | `spam`, `newsletter`, `support`, `sales`, `internal`, `billing`, `security`               |
| `route`    | `inbox`, `archive`, `support_team`, `sales_team`, `security_team`, `billing_team`, `trash`, `human_review` |

The agent responds using XML tags:

```xml
<priority>urgent</priority>
<category>security</category>
<route>security_team</route>
```

## Observation Space

Each observation provides:

| Field                    | Type          | Description                                          |
|--------------------------|---------------|------------------------------------------------------|
| `email_id`               | string        | Unique email identifier                              |
| `email_subject`          | string        | Subject line                                         |
| `email_sender`           | string        | Sender address                                       |
| `email_body`             | string        | Full body text                                       |
| `last_priority_correct`  | bool or null  | Was the previous action's priority correct?          |
| `last_category_correct`  | bool or null  | Was the previous action's category correct?          |
| `last_route_correct`     | bool or null  | Was the previous action's route correct?             |
| `emails_remaining`       | int           | Emails left in this episode (0 = last)               |
| `current_streak`         | int           | Consecutive perfect triage decisions                 |
| `metadata.linked_incident` | bool        | Hint that this email relates to another in the batch |

**Security note:** The observation does NOT contain ground truth for the current email. Ground truth for the previously-graded email is embedded in `metadata.graded_true_*` keys after the agent acts, for client-side grader use only.

---

## Tasks

### Task 1: Spam Detection (Easy)

Binary classification -- is this spam/phishing or legitimate? Includes obvious spam (lottery scams, fake prizes) and sophisticated phishing (CEO impersonation, spoofed invoices). Score: 1.0 for correct, 0.0 for incorrect. Success threshold: 0.6.

### Task 2: Priority Classification (Medium)

Assign the exact urgency level. Requires understanding business context -- a production outage is urgent, a newsletter is low, but a phishing email impersonating the CEO is also urgent because it is an active security threat. Score: 1.0 for exact match, 0.0 otherwise. Success threshold: 0.5.

### Task 3: Full Triage (Hard)

Weighted score across all three dimensions. Includes all environment features: phishing emails, cross-email dependencies (linked incidents get bonus for consistent routing), and dynamic escalation consequences (mishandled urgent emails trigger angry follow-ups with penalty multipliers). Normalized to [0.0, 1.0]. Success threshold: 0.4.

### Task 4: Critical Escalation (Hard)

Identify business-critical emails (legal disputes, GDPR compliance, large contracts, insurance claims, policy changes) and route to `human_review`. Penalizes both missed escalations AND over-escalation equally -- the agent must learn the boundary between routine and critical. Score: 1.0 for correct routing decision, 0.0 otherwise. Success threshold: 0.6.

---

## Reward Design

### Base Score (per email)

| Component        | Weight | Description                                      |
|------------------|--------|--------------------------------------------------|
| Priority correct | +1.0   | Most important signal                            |
| Category correct | +0.5   | Email type identification                        |
| Route correct    | +0.3   | Correct team/queue assignment                    |
| Format bonus     | +0.1   | Priority correct AND >=1 other field correct     |
| Perfect bonus    | +0.2   | All three fields correct                         |
| **Max base**     | **2.1** |                                                 |

### Reward Shaping

| Modifier                | Value    | Trigger                                               |
|-------------------------|----------|-------------------------------------------------------|
| Urgency multiplier      | x0.8-2.0 | Scales base score by true email urgency              |
| Streak bonus            | +0.3     | 3+ consecutive perfect triage decisions               |
| Dependency bonus        | +0.4     | Linked emails routed to the same team consistently    |
| Overload penalty        | -0.5     | Urgent/high email misclassified as low/medium         |
| Escalation multiplier   | /1.5     | Reduced reward on injected angry follow-up emails     |

### Anti-Exploit Protections

- Ground truth is never exposed before the agent acts
- Escalation injections dynamically extend episodes, preventing gaming of fixed-length expectations
- Phishing emails are designed to bypass simple keyword matching

---

## Novel Environment Mechanics

### 1. Phishing Detection

Six sophisticated phishing templates that mimic legitimate communications: CEO wire transfer requests, fake IT password resets, spoofed invoices with changed bank details, fake Google Drive shares, and fraudulent DocuSign links. The agent must identify subtle red flags (suspicious URLs, urgency pressure, unusual requests) rather than relying on obvious spam signals.

### 2. Cross-Email Dependencies

Three dependency clusters (security incident chain, client churn risk chain, compliance chain) where 2 emails reference the same underlying incident. The agent receives a `linked_incident` hint in metadata. Consistent routing of linked emails earns a +0.4 dependency bonus, testing whether the agent can maintain contextual awareness across an episode.

### 3. Escalation Consequences

When the agent misclassifies an urgent/high email as low/medium, an angry follow-up email is dynamically injected 2 positions ahead in the queue. These escalation emails carry a 1.5x penalty multiplier, meaning mistakes on them cost more. This mirrors real-world dynamics where missed urgent emails generate increasingly heated follow-ups.

---

## Setup Instructions

### Local Development

```bash
git clone <repo-url>
cd Email_OpenEnv/Email_RL

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 8000:8000 email-triage-env
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
export EMAIL_RL_SERVER_URL="http://localhost:8000"

python inference.py
```

### Hugging Face Spaces

Deployed as a Docker Space tagged with openenv. Responds to all OpenEnv API endpoints.

---

## API Endpoints

| Endpoint   | Method | Description                                         |
|------------|--------|-----------------------------------------------------|
| `/`        | GET    | Health check and metadata                           |
| `/health`  | GET    | Simple health check                                 |
| `/reset`   | POST   | Reset environment, returns first email observation  |
| `/step`    | POST   | Submit triage action, returns reward + next email   |
| `/state`   | GET    | Current episode state                               |
| `/schema`  | GET    | Action and observation JSON schemas                 |
| `/ws`      | WS     | WebSocket for persistent sessions                   |

---

## Baseline Scores

| Task                     | Difficulty | Model                      | Score  |
|--------------------------|------------|----------------------------|--------|
| spam-detection           | Easy       | Qwen2.5-72B-Instruct      | ~0.85  |
| priority-classification  | Medium     | Qwen2.5-72B-Instruct      | ~0.60  |
| full-triage              | Hard       | Qwen2.5-72B-Instruct      | ~0.50  |
| critical-escalation      | Hard       | Qwen2.5-72B-Instruct      | ~0.65  |

---

## Project Structure

```
Email_RL/
    server/
        __init__.py
        app.py                      # FastAPI server (OpenEnv HTTP + WS)
        Email_RL_environment.py     # Core environment logic + reward shaping
    __init__.py                     # Package exports
    models.py                       # Pydantic typed models (Action, Observation)
    client.py                       # WebSocket client for programmatic use
    inference.py                    # Baseline inference script (4 tasks)
    train.py                        # GRPO training script (optional)
    openenv.yaml                    # OpenEnv metadata spec
    Dockerfile                      # Container definition
    pyproject.toml                  # Package config + dependencies
    .env.example                    # Environment variable template
    README.md                       # This file
```

## Validation

```bash
# Run OpenEnv validator
openenv validate

# Test endpoints
curl http://localhost:8000/
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/state
```