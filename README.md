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

# Email Triage RL -- OpenEnv Environment

An OpenEnv environment where AI agents learn to prioritize, categorize, route, and fully handle business emails for a fictional B2B SaaS company. Goes beyond simple classification -- the agent must generate action plans, detect security threats, and orchestrate cross-system responses like a real operations team.

## Motivation

Email triage is one of the most universal knowledge-work bottlenecks. But classification alone is not enough -- in the real world, triaging an email means deciding what to DO about it. This environment tests whether LLM agents can develop the full operational judgment needed to handle a corporate inbox end-to-end:

- Classify emails by urgency, type, and routing destination
- Generate complete action plans (which systems to trigger, who to notify, what SLA to meet)
- Detect sophisticated social engineering and phishing attacks with structured threat reports
- Handle business-critical escalations requiring human sign-off

Why this matters for the RL/agent community:

- Genuine, high-frequency business task -- not a toy problem
- 6 tasks spanning classification, action planning, and security intelligence
- Multi-dimensional decision space with rich partial-credit reward signals
- Novel mechanics: phishing detection, cross-email dependencies, escalation consequences
- GRPO-compatible reward design for direct RL training

## Environment Description

The agent manages the inbox for Nexora Technologies, a B2B SaaS project management company. Each episode presents 10 emails drawn from a balanced pool of 7 categories, including business-critical emails requiring human sign-off, phishing attempts disguised as legitimate messages, and linked incident chains.

The agent operates at three levels:

1. Classification -- priority, category, and routing (Tasks 1-4)
2. Action Orchestration -- generate a full action plan with systems, stakeholders, and SLA (Task 5)
3. Security Intelligence -- produce a structured threat assessment report (Task 6)

## Action Space

The agent outputs these fields per email:

| Field | Type | Description |
|-------|------|-------------|
| priority | string | low, medium, high, urgent |
| category | string | spam, newsletter, support, sales, internal, billing, security |
| route | string | inbox, archive, support_team, sales_team, security_team, billing_team, trash, human_review |
| action_plan | JSON string (optional) | Action orchestration plan (Task 5 only) |
| threat_report | JSON string (optional) | Security threat assessment (Task 6 only) |

The agent responds using XML tags:

```xml
<priority>urgent</priority>
<category>security</category>
<route>security_team</route>
<action_plan>{"actions": [...], "sla_deadline": "1 hour", "stakeholders_to_notify": [...]}</action_plan>
<threat_report>{"is_threat": true, "threat_type": "ceo_impersonation", "risk_score": 9.2, ...}</threat_report>
```

## Observation Space

Each observation provides:

| Field | Type | Description |
|-------|------|-------------|
| email_id | string | Unique email identifier |
| email_subject | string | Subject line |
| email_sender | string | Sender address |
| email_body | string | Full body text |
| last_priority_correct | bool or null | Was the previous priority correct? |
| last_category_correct | bool or null | Was the previous category correct? |
| last_route_correct | bool or null | Was the previous route correct? |
| emails_remaining | int | Emails left (0 = last) |
| current_streak | int | Consecutive perfect decisions |
| linked_incident | bool | Part of a cross-email cluster |
| is_phishing | bool | Whether this is a phishing attempt (for grader use) |
| is_business_critical | bool | Whether this requires human_review (for grader use) |

## Tasks

### Task 1: Spam Detection (Easy)

Binary classification -- is this spam/phishing or legitimate? Includes obvious spam and sophisticated phishing. Score: 1.0 for correct, 0.0 for incorrect. Success threshold: 0.6.

### Task 2: Priority Classification (Medium)

Assign the exact urgency level. Requires understanding business context and threat severity. Score: 1.0 for exact match, 0.0 otherwise. Success threshold: 0.5.

### Task 3: Full Triage (Hard)

Weighted score across all three classification dimensions (priority, category, route). Includes phishing detection, cross-email dependencies, and escalation consequences. Normalized to [0.0, 1.0]. Success threshold: 0.4.

### Task 4: Critical Escalation (Hard)

Identify business-critical emails (legal disputes, GDPR compliance, large contracts, insurance claims, policy changes) and route to human_review. Penalizes both missed escalations AND over-escalation. Score: 1.0 for correct, 0.0 otherwise. Success threshold: 0.6.

### Task 5: Action Orchestrator (Hard) -- NEW

The agent generates a complete action plan for each email. Graded on:

- Valid JSON with actions list (0.20)
- Appropriate systems for email priority/category (0.25) -- e.g., pagerduty for urgent, jira for bugs
- SLA deadline calibrated to urgency (0.20) -- e.g., "1 hour" for urgent, "end of day" for high
- Relevant stakeholders identified (0.20) -- e.g., CTO for P1, team lead for medium
- Response draft included when needed (0.15)

Success threshold: 0.3.

### Task 6: Threat Assessment (Hard) -- NEW

The agent produces a structured security threat report. Graded on:

- Correct threat/no-threat classification (0.30)
- Attack vector identified for phishing emails (0.20) -- e.g., "ceo_impersonation", "credential_phishing"
- Quality of indicators list (0.20) -- e.g., "spoofed_domain", "urgency_pressure", "secrecy_pressure"
- Sensible recommended actions (0.15) -- e.g., "quarantine", "notify_security", "verify_with_sender"
- Risk score calibrated to actual threat level (0.15)

Success threshold: 0.3.

## Reward Design

### Classification Tasks (1-4)

Base Score per email: Priority correct (+1.0), Category correct (+0.5), Route correct (+0.3), Format bonus (+0.1), Perfect bonus (+0.2). Max base: 2.1.

Reward Shaping: Urgency multiplier (x0.8-2.0), Streak bonus (+0.3), Dependency bonus (+0.4), Overload penalty (-0.5), Escalation multiplier (/1.5).

### Orchestrator Task (5)

Multi-dimensional grading across 5 criteria: structure (0.20), system selection (0.25), SLA calibration (0.20), stakeholder identification (0.20), response quality (0.15). Total normalized to [0.0, 1.0].

### Threat Assessment Task (6)

Multi-dimensional grading across 5 criteria: threat classification (0.30), attack vector (0.20), indicators (0.20), recommended actions (0.15), risk score calibration (0.15). Total normalized to [0.0, 1.0].

## Novel Environment Mechanics

### 1. Phishing Detection

Six sophisticated phishing templates mimicking real communications: CEO wire transfer requests, fake IT password resets, spoofed invoices with changed bank details, fake Google Drive shares, and fraudulent DocuSign links.

### 2. Cross-Email Dependencies

Three dependency clusters (security incident chain, client churn risk chain, compliance chain) where 2 emails reference the same underlying incident. Consistent routing earns a +0.4 dependency bonus.

### 3. Escalation Consequences

When the agent misclassifies an urgent/high email as low/medium, an angry follow-up email is dynamically injected 2 positions ahead with a 1.5x penalty multiplier.

### 4. Action Orchestration

The agent must reason about which external systems to trigger (PagerDuty, Slack, Jira, CRM, Calendar), calibrate SLA deadlines to email urgency, and identify the right stakeholders to notify.

### 5. Security Intelligence

The agent must detect social engineering attacks, identify specific attack vectors and red-flag indicators, recommend defensive actions, and calibrate a risk score -- going beyond binary spam detection to structured threat analysis.

## Anti-Exploit Protections

- Ground truth is never exposed before the agent acts
- Escalation injections dynamically extend episodes
- Phishing emails bypass simple keyword matching
- Action plan grader validates semantic correctness, not just structure
- Threat assessment grader requires calibrated confidence and risk scores

## Setup Instructions

### Local Development

```bash
git clone <repo-url>
cd Email_OpenEnv/Email_RL
uv sync
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

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check and metadata |
| /health | GET | Simple health check |
| /reset | POST | Reset environment, returns first email |
| /step | POST | Submit triage action, returns reward |
| /state | GET | Current episode state |
| /schema | GET | Action and observation schemas |
| /ws | WS | WebSocket for persistent sessions |

## Baseline Scores

| Task | Difficulty | Model | Score |
|------|-----------|-------|-------|
| spam-detection | Easy | Qwen2.5-72B-Instruct | 0.990 |
| priority-classification | Medium | Qwen2.5-72B-Instruct | 0.500 |
| full-triage | Hard | Qwen2.5-72B-Instruct | 0.648 |
| critical-escalation | Hard | Qwen2.5-72B-Instruct | 0.794 |
| action-orchestrator | Hard | Qwen2.5-72B-Instruct | 0.669 |
| threat-assessment | Hard | Qwen2.5-72B-Instruct | 0.340 |

## Project Structure

```
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
```

## Validation

```bash
openenv validate
curl http://localhost:8000/
curl -X POST http://localhost:8000/reset
curl http://localhost:8000/state
```