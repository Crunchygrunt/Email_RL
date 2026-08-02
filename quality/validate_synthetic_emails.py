"""
Layer 1 of the synthetic-email data quality gate: structural + statistical
checks on the template pool in server/Email_RL_environment.py, run
standalone or in CI.

    python quality/validate_synthetic_emails.py

Imports the real template lists and generation functions directly from
Email_RL_environment.py -- no duplicated data, zero drift risk (same
principle as run_episodes.py importing inference.py's TASKS/graders
verbatim).

Two other layers exist alongside this one:
  - Layer 2 (runtime): `_check_email_quality()` in Email_RL_environment.py
    checks invariants on each email actually served at runtime and logs
    the result as `email_quality_flags` in telemetry.
  - Layer 3 (warehouse): dbt tests in warehouse/models/staging/schema.yml
    and warehouse/tests/*.sql check the same invariants against the
    aggregate collected dataset, after the fact.
This script is the earliest and cheapest of the three -- it never touches
the network, the server, or an LLM, so it's fast enough to run on every
commit.

Exit code 0 if every check passes, 1 otherwise (CI-friendly).
"""

from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

# --- import wiring -----------------------------------------------------
# Mirrors the dual-mode (package vs. flat) import pattern already used by
# Email_RL_environment.py / client.py / run_episodes.py in this project.
# Not yet verified against the real repo's exact directory layout -- please
# confirm this resolves once dropped in; if the package-relative import
# fails, the flat fallback below should still work as long as this script
# is run with the project root as the working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_PACKAGE_DIR)
sys.path.insert(0, _PACKAGE_DIR)
sys.path.insert(0, os.path.join(_PACKAGE_DIR, "server"))
sys.path.insert(0, _REPO_ROOT)

try:
    from Email_RL.server import Email_RL_environment as env_mod
    from Email_RL import models
except ImportError:
    import server.Email_RL_environment as env_mod
    import models


VALID_PLACEHOLDER_KEYS = {"name", "product", "plan", "day", "amount", "id"}

# Minimum fraction of sampled episodes a standard category must appear in
# before it's considered "systematically missing" (regression guard for
# the bug fixed in Email_RL_environment.py: _sample_episode() used to
# always drop the same category -- see WAREHOUSE.md). Expected rate is
# ~6/7 (~85.7%) since exactly one of 7 standard slots is dropped per
# episode; this threshold is deliberately loose so normal sampling
# variance never trips it, while a near-total omission (the original bug
# produced 0%) fails hard.
MIN_CATEGORY_EPISODE_RATE = 0.5

# Sender-domain "suspicious TLD" signal must be near-exclusive to phishing
# emails (regression guard for the sender-domain-pool-overlap bug also
# fixed in Email_RL_environment.py, which produced ~40%/40% noise).
MIN_PHISHING_SUSPICIOUS_RATE = 0.9
MAX_LEGIT_SUSPICIOUS_RATE = 0.05

N_SAMPLE_EPISODES = 500
N_SENDER_SAMPLES = 3000


class CheckFailure(Exception):
    pass


def _fail(msg: str) -> None:
    raise CheckFailure(msg)


# ---------------------------------------------------------------------
# Structural checks (static -- on the template definitions themselves)
# ---------------------------------------------------------------------

def check_category_coverage() -> str:
    by_cat: Dict[str, List[int]] = defaultdict(list)
    for _, _, _, cat in env_mod._EMAIL_TEMPLATES:
        by_cat[cat].append(1)
    missing = [c for c in models.CATEGORIES if not by_cat[c]]
    if missing:
        _fail(f"categories with zero standard templates: {missing}")
    return f"all {len(models.CATEGORIES)} categories have >=1 standard template"


def _iter_template_texts() -> List[Tuple[str, int, str, str]]:
    """(pool_name, index, subject, body) for every template across all pools."""
    out: List[Tuple[str, int, str, str]] = []
    for i, (subj, body, _, _) in enumerate(env_mod._EMAIL_TEMPLATES):
        out.append(("standard", i, subj, body))
    for i, (subj, body, _, _) in enumerate(env_mod._CRITICAL_EMAIL_TEMPLATES):
        out.append(("critical", i, subj, body))
    for i, (subj, body, _, _) in enumerate(env_mod._PHISHING_EMAIL_TEMPLATES):
        out.append(("phishing", i, subj, body))
    for i, (subj, body, _, _) in enumerate(env_mod._ESCALATION_TEMPLATES):
        out.append(("escalation", i, subj, body))
    for ci, cluster in enumerate(env_mod._DEPENDENCY_CLUSTERS):
        for ei, entry in enumerate(cluster):
            out.append((f"cluster_{ci}", ei, entry[0], entry[1]))
    return out


def check_placeholder_keys() -> str:
    templates = _iter_template_texts()
    bad = []
    for pool, idx, subj, body in templates:
        for text in (subj, body):
            keys = set(re.findall(r"\{(\w+)\}", text))
            unknown = keys - VALID_PLACEHOLDER_KEYS
            if unknown:
                bad.append(f"{pool}[{idx}]: unknown placeholder(s) {unknown} in {text[:60]!r}")
    if bad:
        _fail("; ".join(bad))
    return f"all {len(templates)} templates use only valid placeholder keys {sorted(VALID_PLACEHOLDER_KEYS)}"


def check_enum_validity() -> str:
    bad = []

    def _check(pool: str, items) -> None:
        for i, item in enumerate(items):
            prio, cat = item[2], item[3]
            if prio not in models.PRIORITIES:
                bad.append(f"{pool}[{i}]: invalid priority {prio!r}")
            if cat not in models.CATEGORIES:
                bad.append(f"{pool}[{i}]: invalid category {cat!r}")

    _check("standard", env_mod._EMAIL_TEMPLATES)
    _check("critical", env_mod._CRITICAL_EMAIL_TEMPLATES)
    _check("phishing", env_mod._PHISHING_EMAIL_TEMPLATES)
    _check("escalation", env_mod._ESCALATION_TEMPLATES)
    for ci, cluster in enumerate(env_mod._DEPENDENCY_CLUSTERS):
        _check(f"cluster_{ci}", cluster)

    if bad:
        _fail("; ".join(bad))
    return "all templates reference valid priority/category enum values"


def check_no_duplicate_subjects() -> str:
    subjects = [subj for subj, _, _, _ in env_mod._EMAIL_TEMPLATES]
    dupes = {s: n for s, n in Counter(subjects).items() if n > 1}
    if dupes:
        _fail(f"duplicate standard-template subjects: {dupes}")
    return "no duplicate subject templates in the standard pool"


# ---------------------------------------------------------------------
# Statistical checks (sample _sample_episode() many times)
# ---------------------------------------------------------------------

class _FakeEnvForSampling:
    """Just enough of EmailTriageEnvironment for _sample_episode() to run
    standalone, without needing a real openenv server/session."""
    EPISODE_LENGTH = env_mod.EmailTriageEnvironment.EPISODE_LENGTH
    _sample_episode = env_mod.EmailTriageEnvironment._sample_episode


def _sample_many(n: int) -> List[List[Dict[str, Any]]]:
    fake = _FakeEnvForSampling()
    return [fake._sample_episode() for _ in range(n)]


def check_episode_composition(episodes: List[List[Dict[str, Any]]]) -> str:
    expected_len = env_mod.EmailTriageEnvironment.EPISODE_LENGTH
    bad = []
    for i, ep in enumerate(episodes):
        if len(ep) != expected_len:
            bad.append(f"episode {i}: length {len(ep)} != {expected_len}")
            continue
        n_phish = sum(1 for e in ep if e["is_phishing"])
        n_crit = sum(1 for e in ep if e["is_business_critical"])
        n_clust = sum(1 for e in ep if e.get("cluster_id"))
        if n_phish != 1:
            bad.append(f"episode {i}: {n_phish} phishing emails (expected 1)")
        if n_crit != 1:
            bad.append(f"episode {i}: {n_crit} critical emails (expected 1)")
        if n_clust != 2:
            bad.append(f"episode {i}: {n_clust} cluster emails (expected 2)")
    if bad:
        _fail(f"{len(bad)} violation(s), e.g.: " + "; ".join(bad[:5]))
    return f"all {len(episodes)} sampled episodes have exactly 1 phishing + 1 critical + 2 clustered email(s)"


def check_cluster_pairing(episodes: List[List[Dict[str, Any]]]) -> str:
    bad = []
    for i, ep in enumerate(episodes):
        clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in ep:
            if e.get("cluster_id"):
                clusters[e["cluster_id"]].append(e)
        for cid, members in clusters.items():
            if len(members) != 2:
                bad.append(f"episode {i}: cluster {cid} has {len(members)} member(s) (expected 2)")
    if bad:
        _fail("; ".join(bad[:5]))
    return "every dependency cluster appears as exactly 2 linked emails per episode"


def check_category_balance(episodes: List[List[Dict[str, Any]]]) -> str:
    n = len(episodes)
    category_episode_hits = {c: 0 for c in models.CATEGORIES}
    for ep in episodes:
        seen = set()
        for e in ep:
            if not e["is_phishing"] and not e["is_business_critical"] and not e.get("cluster_id"):
                seen.add(e["category"])
        for c in seen:
            category_episode_hits[c] += 1

    bad = []
    for c, hits in category_episode_hits.items():
        rate = hits / n
        if rate < MIN_CATEGORY_EPISODE_RATE:
            bad.append(f"{c}: appeared in only {hits}/{n} episodes ({rate:.1%})")
    if bad:
        _fail("; ".join(bad))

    rates = ", ".join(f"{c}={category_episode_hits[c] / n:.0%}" for c in models.CATEGORIES)
    return f"all standard categories appear in >={MIN_CATEGORY_EPISODE_RATE:.0%} of episodes ({rates})"


def check_phishing_sender_signal() -> str:
    phish_hits = 0
    legit_hits = 0
    for _ in range(N_SENDER_SAMPLES):
        e = env_mod._generate_email(phishing=True)
        dom = e["sender"].split("@")[-1]
        if any(dom.endswith(t) for t in env_mod._SUSPICIOUS_TLDS):
            phish_hits += 1
    for _ in range(N_SENDER_SAMPLES):
        e = env_mod._generate_email(critical=False)
        dom = e["sender"].split("@")[-1]
        if any(dom.endswith(t) for t in env_mod._SUSPICIOUS_TLDS):
            legit_hits += 1

    phish_rate = phish_hits / N_SENDER_SAMPLES
    legit_rate = legit_hits / N_SENDER_SAMPLES

    if phish_rate < MIN_PHISHING_SUSPICIOUS_RATE:
        _fail(
            f"only {phish_rate:.1%} of phishing emails hit the suspicious-TLD "
            f"sender check (expected >={MIN_PHISHING_SUSPICIOUS_RATE:.0%})"
        )
    if legit_rate > MAX_LEGIT_SUSPICIOUS_RATE:
        _fail(
            f"{legit_rate:.1%} of LEGITIMATE emails hit the suspicious-TLD sender "
            f"check (expected <={MAX_LEGIT_SUSPICIOUS_RATE:.0%}) -- sender domain "
            f"pools are overlapping again"
        )
    return f"phishing sender-domain signal is real: phishing={phish_rate:.0%} suspicious vs. legit={legit_rate:.0%}"


def report_template_reuse(episodes: List[List[Dict[str, Any]]]) -> str:
    """Informational only, not pass/fail: reports template_idx reuse
    frequency per pool across sampled episodes so skew is visible even
    though nothing here treats it as a hard failure yet."""
    pool_counts: Dict[str, Counter] = defaultdict(Counter)
    for ep in episodes:
        for e in ep:
            pool = e.get("template_pool")
            idx = e.get("template_idx")
            if pool and idx is not None:
                pool_counts[pool][idx] += 1
    if not pool_counts:
        return "no template_idx data available (older Email_RL_environment.py without provenance tracking?)"
    lines = []
    for pool, counts in sorted(pool_counts.items()):
        min_c, max_c = min(counts.values()), max(counts.values())
        lines.append(f"{pool}: {len(counts)} templates used, reuse count range {min_c}-{max_c}")
    return "; ".join(lines)


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def main() -> int:
    print(f"Sampling {N_SAMPLE_EPISODES} episodes for statistical checks...")
    episodes = _sample_many(N_SAMPLE_EPISODES)

    checks: List[Tuple[str, Any, tuple]] = [
        ("category coverage", check_category_coverage, ()),
        ("placeholder keys", check_placeholder_keys, ()),
        ("enum validity", check_enum_validity, ()),
        ("no duplicate subjects", check_no_duplicate_subjects, ()),
        ("episode composition", check_episode_composition, (episodes,)),
        ("cluster pairing", check_cluster_pairing, (episodes,)),
        ("category balance", check_category_balance, (episodes,)),
        ("phishing sender signal", check_phishing_sender_signal, ()),
    ]

    n_failed = 0
    print("\n=== Synthetic Email Data Quality Gate (Layer 1) ===\n")
    for name, fn, args in checks:
        try:
            msg = fn(*args)
            print(f"[PASS] {name}: {msg}")
        except CheckFailure as exc:
            n_failed += 1
            print(f"[FAIL] {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- surface unexpected errors, don't hide them
            n_failed += 1
            print(f"[ERROR] {name}: unexpected exception: {exc}")

    print("\n--- informational (not pass/fail) ---")
    print(f"template reuse: {report_template_reuse(episodes)}")

    print(f"\n{len(checks) - n_failed}/{len(checks)} checks passed.")
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
