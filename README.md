# Email Triage RL Environment

A real-world reinforcement learning environment where agents learn to triage business emails - assigning priority, category, and routing decisions across a synthetic inbox of 30 plus email templates spanning 7 categories.

Built on the OpenEnv framework and deployed as a HuggingFace Space.

---

## Motivation

Email triage is a genuine daily task in every B2B company. A skilled human triager must:

* Distinguish urgent production incidents from newsletters
* Identify business-critical emails such as legal disputes and contract negotiations that require human sign-off
* Route each email to the right team without overwhelming any single queue
* Spot phishing and social engineering attempts hidden among legitimate mail

This environment models that task faithfully, making it suitable for evaluating and training language model based agents on real-world classification under uncertainty.

---

## Environment Description

Each episode consists of 10 emails drawn from a synthetic dataset balanced across all 7 categories. The agent receives one email per step and must classify it along three axes simultaneously.

The server runs as a FastAPI application. In stateless HTTP mode, each reset and step pair is self-contained. The server automatically resets when a step arrives without a prior reset on the same instance.

---

## Action Space

Type: EmailTriageAction (Pydantic model)

Fields:

* priority: low, medium, high, urgent
* category: spam, newsletter, support, sales, internal, billing, security
* route: inbox, archive, support_team, sales_team, security_team, billing_team, trash, human_review

Routing rules:

* human_review is reserved for business-critical emails requiring human sign-off
* All other routes map to their category (example: spam to trash, newsletter to archive)

---

## Observation Space

Type: EmailTriageObservation (Pydantic model)

Fields include:

* email_id
* email_subject
* email_sender
* email_body
* last_priority_correct
* last_category_correct
* last_route_correct
* emails_remaining
* current_streak
* done
* reward
* metadata

Metadata includes:

* true_priority
* true_category
* true_route
* is_business_critical
* graded_true_priority
* graded_true_category
* graded_true_route
* graded_is_business_critical
* streak
* episode_id

---

## Reward Function

shaped_reward = base_score multiplied by urgency_multiplier plus streak_bonus minus overload_penalty

Where:

* base_score is calculated using correctness of priority, category, and route along with small bonuses
* urgency_multiplier depends on priority level
* streak_bonus is applied after multiple correct steps
* overload_penalty is applied when urgent emails are misclassified as low priority

This creates a dense reward signal across the episode.

---

## Tasks

Task 1: spam-detection (Easy)

* Objective: classify email as spam or not
* Output: 1.0 if correct, otherwise 0.0

Task 2: priority-classification (Medium)

* Objective: assign correct urgency level
* Strict matching required

Task 3: full-triage (Hard)

* Objective: correctly classify priority, category, and route together
* Weighted scoring system

Task 4: critical-escalation (Hard)

* Objective: detect business-critical emails and route to human_review
* Penalizes both under and over escalation

---

## Email Dataset

Includes 30 plus synthetic templates across:

* spam
* newsletter
* support
* sales
* internal
* billing
* security
* business-critical emails

Templates use randomized values to ensure variety.

---

## Setup and Usage

Install dependencies:
pip install openenv-core httpx python-dotenv openai

Environment variables:
OPENAI_API_KEY
API_BASE_URL
MODEL_NAME
EMAIL_RL_SERVER_URL

Run server:
uvicorn server.app:app --host 0.0.0.0 --port 8000

Run inference:
python inference.py

Run validator:
openenv validate

---

## Baseline Notes

Performance varies depending on model size and randomness. Larger instruction-tuned models perform significantly better.

---

## Project Structure

Email_RL/

* inference.py
* models.py
* client.py
* openenv.yaml
* server/

  * Email_RL_environment.py
  * app.py

---

## OpenEnv Compliance

* reset returns first observation
* step returns next observation, reward, and done flag
* state includes episode tracking
* Pydantic models used for typing
* Compatible with HTTP deployment
