"""
train.py — GRPO training for the Email Triage RL Environment
=============================================================
Connects the EmailTriageEnv reward signal to TRL's GRPOTrainer.

Usage:
    python train.py
    python train.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300
    python train.py --model meta-llama/Llama-3.2-1B-Instruct --batch 8
"""

import argparse
import asyncio
import re
import textwrap
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.grpo_config import GRPOConfig

from Email_RL import EmailTriageAction, EmailTriageEnv
from Email_RL.models import CATEGORIES, PRIORITIES, ROUTES

# ── CLI args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",    default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--server",   default="http://localhost:8000")
parser.add_argument("--steps",    type=int, default=300)
parser.add_argument("--batch",    type=int, default=4)
parser.add_argument("--lr",       type=float, default=5e-6)
parser.add_argument("--output",   default="./email_triage_grpo")
args = parser.parse_args()

MODEL_NAME  = args.model
SERVER_URL  = args.server
MAX_STEPS   = args.steps
BATCH_SIZE  = args.batch

# ── Prompts (identical to Inference.py) ───────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert email triage assistant for a B2B software company.
    You will be shown a business email and must classify it.

    PRIORITY : low | medium | high | urgent
    CATEGORY : spam | newsletter | support | sales | internal | billing | security
    ROUTE    : inbox | archive | support_team | sales_team |
               security_team | billing_team | trash

    REWARD STRUCTURE:
        Correct priority  → +1.0   Correct category → +0.5   Correct route → +0.3
        All correct       → +0.2 bonus
        Urgent/high misclassified as low/medium → −0.5 penalty
        3+ consecutive perfect → +0.3 streak bonus

    Reply ONLY with these three XML tags:
        <priority>VALUE</priority>
        <category>VALUE</category>
        <route>VALUE</route>
""").strip()


def build_prompt(subject: str, sender: str, body: str) -> str:
    return (
        f"From   : {sender}\n"
        f"Subject: {subject}\n"
        f"Body   :\n{body}\n\n"
        "Classify this email. Reply ONLY with the three XML tags."
    )


# ── XML parser ─────────────────────────────────────────────────────────────
_P = re.compile(r"<priority>\s*([^<]+?)\s*</priority>", re.I)
_C = re.compile(r"<category>\s*([^<]+?)\s*</category>", re.I)
_R = re.compile(r"<route>\s*([^<]+?)\s*</route>",       re.I)


def parse_action(text: str) -> EmailTriageAction:
    priority = _P.search(text)
    category = _C.search(text)
    route    = _R.search(text)
    return EmailTriageAction(
        priority = priority.group(1).strip().lower() if priority else "low",
        category = category.group(1).strip().lower() if category else "spam",
        route    = route.group(1).strip().lower()    if route    else "trash",
    )


# ── Dataset ────────────────────────────────────────────────────────────────
# GRPOTrainer requires a HuggingFace Dataset with a "prompt" column.
# We generate a large pool of prompts by running resets against the server.

async def _collect_prompts(n: int) -> List[Dict]:
    """Hit the environment server to collect n email prompts."""
    rows = []
    async with EmailTriageEnv(base_url=SERVER_URL) as env:
        while len(rows) < n:
            result = await env.reset()
            obs    = result.observation
            rows.append({
                "prompt": [
                    {"role": "system",  "content": SYSTEM_PROMPT},
                    {"role": "user",    "content": build_prompt(
                        obs.email_subject, obs.email_sender, obs.email_body
                    )},
                ],
                # Carry ground truth for the reward function
                "_true_priority": obs.metadata.get("true_priority", ""),
                "_true_category": obs.metadata.get("true_category", ""),
                "_true_route":    obs.metadata.get("true_route", ""),
            })
    return rows


def build_dataset(n_prompts: int = 500) -> Dataset:
    print(f"Collecting {n_prompts} prompts from {SERVER_URL} …")
    rows = asyncio.run(_collect_prompts(n_prompts))
    return Dataset.from_list(rows)


# ── Reward function ────────────────────────────────────────────────────────
# GRPOTrainer calls this with a batch of completions.
# We reconstruct the shaped reward locally using the same formula as the env
# so we don't need a live async call per completion during training.

from Email_RL.server.Email_RL_environment import TriageGrader
from Email_RL.models import URGENCY_BONUS

_grader = TriageGrader()
STREAK_BONUS    = 0.3
OVERLOAD_PENALTY = 0.5


def triage_reward(
    completions: List[str],
    _true_priority: List[str],
    _true_category: List[str],
    _true_route:    List[str],
    **kwargs,
) -> List[float]:
    """
    Reward function for GRPOTrainer.

    Mirrors the shaped reward in EmailTriageEnvironment.step() exactly:
        base_score × urgency_multiplier − overload_penalty
    (streak bonus is omitted here since GRPO samples are i.i.d.)
    """
    rewards = []
    for text, tp, tc, tr in zip(completions, _true_priority, _true_category, _true_route):
        action = parse_action(text)
        grade  = _grader.grade(
            action={"priority": action.priority,
                    "category": action.category,
                    "route":    action.route},
            email ={"priority": tp,
                    "category": tc,
                    "route":    tr},
        )
        urgency_mult  = URGENCY_BONUS.get(tp, 1.0)
        shaped        = grade.base_score * urgency_mult

        # Overload penalty
        if tp in ("urgent", "high") and action.priority in ("low", "medium"):
            shaped -= OVERLOAD_PENALTY

        rewards.append(round(shaped, 4))
    return rewards


# ── Model + LoRA ───────────────────────────────────────────────────────────
def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,          # remove if you have enough VRAM
    )
    return model, tokenizer


LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


# ── GRPO training config ───────────────────────────────────────────────────
GRPO_CONFIG = GRPOConfig(
    output_dir              = args.output,
    num_train_epochs        = 1,
    max_steps               = MAX_STEPS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 2,
    learning_rate           = args.lr,
    bf16                    = True,
    logging_steps           = 5,
    save_steps              = 50,
    report_to               = "none",       # swap to "wandb" if you want tracking
    # GRPO-specific
    num_generations         = 4,            # completions sampled per prompt
    max_completion_length   = 80,           # XML response is short
    temperature             = 0.7,
    beta                    = 0.01,         # KL penalty coefficient
)


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset           = build_dataset(n_prompts=max(500, MAX_STEPS * BATCH_SIZE))
    model, tokenizer  = load_model_and_tokenizer()

    trainer = GRPOTrainer(
        model          = model,
        args           = GRPO_CONFIG,
        train_dataset  = dataset,
        reward_funcs   = triage_reward,
        peft_config    = LORA_CONFIG,
        processing_class = tokenizer,
    )

    print("\n=== Starting GRPO training ===")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Steps   : {MAX_STEPS}")
    print(f"  Batch   : {BATCH_SIZE} × 4 generations = {BATCH_SIZE * 4} completions/step")
    print(f"  Output  : {args.output}\n")

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\nModel saved to {args.output}")