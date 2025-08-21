#!/usr/bin/env python3
"""
Seed eval tasks data into a running app_backend for manual /tasks/all testing.

Usage:
  python seed_eval_tasks.py \
    --backend-url http://localhost:8000 \
    --user-email seeder@example.com \
    --policies 5 \
    --tasks-per-policy 8
"""

import argparse
import random
import string
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx


# Pools for varied descriptions and tags
_DESC_ADJ = [
    "robust",
    "experimental",
    "baseline",
    "aggressive",
    "conservative",
    "stochastic",
    "deterministic",
    "noisy",
    "optimized",
    "fallback",
    "memory-heavy",
    "speed-focused",
]

_DESC_NOUN = [
    "navigation",
    "memory",
    "arena",
    "wfc",
    "planner",
    "collector",
    "evaluator",
    "learner",
    "agent",
    "policy",
]

_TAG_POOL = [
    "seed",
    "eval_tasks",
    "nightly",
    "ci",
    "perf",
    "smoke",
    "baseline",
    "experimental",
    "regression",
    "stable",
    "alpha",
    "beta",
    "release",
    "hotfix",
    "longrun",
    "gpu",
    "cpu",
    "linux",
    "macos",
    "windows",
    "docker",
    "k8s",
    "retry",
    "priority-low",
    "priority-medium",
    "priority-high",
    "team-a",
    "team-b",
    "team-c",
    "nav",
    "arena",
    "memory",
    "wfc",
]


def _random_description(kind: str) -> str:
    adj = random.choice(_DESC_ADJ)
    noun = random.choice(_DESC_NOUN)
    return f"{kind} - {adj} {noun} {_now_suffix()}"


def _random_tags(min_tags: int = 2, max_tags: int = 5, extra: Optional[List[str]] = None) -> List[str]:
    k = random.randint(min_tags, max_tags)
    picked = set(random.sample(_TAG_POOL, k))
    for t in (extra or []):
        picked.add(t)
    return sorted(picked)


def _rand_commit_hex(n: int = 40) -> str:
    return "".join(random.choices("0123456789abcdef", k=n))


def _now_suffix() -> str:
    return str(int(time.time() * 1000))


def _post_json(
    client: httpx.Client,
    base_url: str,
    path: str,
    json: Dict[str, Any],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["X-Auth-Token"] = token
    r = client.post(f"{base_url}{path}", json=json, headers=headers)
    r.raise_for_status()
    return r.json()


def _get_json(
    client: httpx.Client,
    base_url: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    headers = {}
    if token:
        headers["X-Auth-Token"] = token
    r = client.get(f"{base_url}{path}", params=params or {}, headers=headers)
    r.raise_for_status()
    return r.json()


def create_machine_token(client: httpx.Client, base_url: str, user_email: str) -> str:
    r = client.post(
        f"{base_url}/tokens",
        json={"name": f"seed_{_now_suffix()}", "permissions": ["read", "write"]},
        headers={"X-Auth-Request-Email": user_email},
    )
    r.raise_for_status()
    return r.json()["token"]


def create_training_run(client: httpx.Client, base_url: str, token: str, name: str) -> uuid.UUID:
    description = _random_description("run")
    tags = _random_tags(extra=["seed", "eval_tasks"])
    data = _post_json(
        client,
        base_url,
        "/stats/training-runs",
        json={
            "name": name,
            "attributes": {"source": "seed_script", "group": name.split("_")[0]},
            "url": f"https://example.com/run/{_now_suffix()}",
            "description": description,
            "tags": tags,
        },
        token=token,
    )
    return uuid.UUID(data["id"])


def create_epoch(
    client: httpx.Client, base_url: str, token: str, run_id: uuid.UUID, start: int, end: int
) -> uuid.UUID:
    data = _post_json(
        client,
        base_url,
        f"/stats/training-runs/{run_id}/epochs",
        json={
            "start_training_epoch": start,
            "end_training_epoch": end,
            "attributes": {"lr": "0.001", "notes": "seed"},
        },
        token=token,
    )
    return uuid.UUID(data["id"])


def create_policy(
    client: httpx.Client,
    base_url: str,
    token: str,
    name: str,
    epoch_id: Optional[uuid.UUID],
) -> uuid.UUID:
    description = _random_description("policy")
    data = _post_json(
        client,
        base_url,
        "/stats/policies",
        json={"name": name, "description": description, "url": None, "epoch_id": str(epoch_id) if epoch_id else None},
        token=token,
    )
    return uuid.UUID(data["id"])


def create_task(
    client: httpx.Client,
    base_url: str,
    token: str,
    policy_id: uuid.UUID,
    sim_suite: str,
    attributes: Dict[str, Any],
) -> uuid.UUID:
    # Attributes may already include git_hash; the backend will keep it
    data = _post_json(
        client,
        base_url,
        "/tasks",
        json={
            "policy_id": str(policy_id),
            "sim_suite": sim_suite,
            "attributes": attributes,
        },
        token=token,
    )
    return uuid.UUID(data["id"])


def claim_tasks(
    client: httpx.Client, base_url: str, token: str, task_ids: List[uuid.UUID], assignee: str
) -> List[uuid.UUID]:
    if not task_ids:
        return []
    data = _post_json(
        client,
        base_url,
        "/tasks/claim",
        json={"tasks": [str(t) for t in task_ids], "assignee": assignee},
        token=token,
    )
    return [uuid.UUID(t) for t in data.get("claimed", [])]


def update_task_statuses(
    client: httpx.Client,
    base_url: str,
    token: str,
    updates: Dict[uuid.UUID, Dict[str, Any]],
    require_assignee: Optional[str] = None,
) -> Dict[uuid.UUID, str]:
    body_updates = {str(k): v for k, v in updates.items()}
    body: Dict[str, Any] = {"updates": body_updates}
    if require_assignee:
        body["require_assignee"] = require_assignee
    data = _post_json(client, base_url, "/tasks/claimed/update", json=body, token=token)
    return {uuid.UUID(k): v for k, v in data.get("statuses", {}).items()}


def seed(
    backend_url: str,
    user_email: str,
    num_policies: int,
    tasks_per_policy: int,
    assignees: List[str],
    sim_suites: List[str],
    git_hashes: List[str],
    seed_value: Optional[int] = None,
) -> None:
    if seed_value is not None:
        random.seed(seed_value)

    with httpx.Client(timeout=30.0) as client:
        token = create_machine_token(client, backend_url, user_email)

        all_policy_ids: List[uuid.UUID] = []
        all_task_ids: List[uuid.UUID] = []

        # Create runs, epochs, policies
        for p in range(num_policies):
            run_name = f"seed_run_{_now_suffix()}_{p}"
            run_id = create_training_run(client, backend_url, token, run_name)
            epoch1 = create_epoch(client, backend_url, token, run_id, 0, 100)
            epoch2 = create_epoch(client, backend_url, token, run_id, 100, 200)

            policy_name = f"seed_policy_{p}_{_now_suffix()}"
            policy_id = create_policy(client, backend_url, token, policy_name, epoch1 if p % 2 == 0 else epoch2)
            all_policy_ids.append(policy_id)

            # Create tasks for this policy with varied attributes
            for t in range(tasks_per_policy):
                suite = sim_suites[(p + t) % len(sim_suites)]
                gh = git_hashes[(p * tasks_per_policy + t) % len(git_hashes)]
                attrs = {
                    "git_hash": gh,
                    "priority": random.choice(["low", "medium", "high"]),
                    "workers_spawned": random.randint(0, 5),
                    "note": f"seed_{p}_{t}",
                }
                task_id = create_task(client, backend_url, token, policy_id, suite, attrs)
                all_task_ids.append(task_id)

        # Claim a subset of tasks across assignees
        random.shuffle(all_task_ids)
        to_claim = all_task_ids[: int(len(all_task_ids) * 0.6)]  # 60% claimed
        claimed_by: Dict[str, List[uuid.UUID]] = {a: [] for a in assignees}

        # Batch claim per assignee
        for i, chunk_start in enumerate(range(0, len(to_claim), 10)):
            chunk = to_claim[chunk_start : chunk_start + 10]
            assignee = assignees[i % len(assignees)]
            claimed = claim_tasks(client, backend_url, token, chunk, assignee)
            claimed_by[assignee].extend(claimed)

        # Update some claimed tasks to done/error, leave some claimed as unprocessed
        for assignee, tasks in claimed_by.items():
            if not tasks:
                continue
            random.shuffle(tasks)
            n = len(tasks)
            to_done = tasks[: n // 3]
            to_error = tasks[n // 3 : 2 * n // 3]
            # leave last third as claimed (unprocessed)

            if to_done:
                updates = {
                    t: {
                        "status": "done",
                        "clear_assignee": random.choice([True, False]),
                        "attributes": {"runtime_sec": round(random.uniform(5, 120), 2), "result": "ok"},
                    }
                    for t in to_done
                }
                update_task_statuses(client, backend_url, token, updates, require_assignee=assignee)

            if to_error:
                updates = {
                    t: {
                        "status": "error",
                        "clear_assignee": random.choice([True, False]),
                        "attributes": {"runtime_sec": round(random.uniform(1, 60), 2), "error": "Timeout"},
                    }
                    for t in to_error
                }
                update_task_statuses(client, backend_url, token, updates, require_assignee=assignee)

        # Optionally mark a few unclaimed tasks as canceled to broaden status mix
        remaining_unclaimed = [t for t in all_task_ids if t not in set(to_claim)]
        to_cancel = remaining_unclaimed[: max(1, len(remaining_unclaimed) // 10)]
        if to_cancel:
            updates = {
                t: {"status": "canceled", "clear_assignee": True, "attributes": {"reason": "seed_cancel"}}
                for t in to_cancel
            }
            update_task_statuses(client, backend_url, token, updates, require_assignee=None)

        # Print a brief summary
        print(f"Seed complete.")
        print(f"- Policies: {len(all_policy_ids)}")
        print(f"- Tasks total: {len(all_task_ids)}")
        print(f"- Claimed: {len(to_claim)}; Unclaimed: {len(all_task_ids) - len(to_claim)}")
        print(f"- Done/Error/Canceled mixed across claimed and unclaimed tasks.")
        print()
        print("Example queries you can run now:")
        print(f"  GET {backend_url}/tasks/all?limit=100")
        print(f"  GET {backend_url}/tasks/all?statuses=done&statuses=error&limit=100")
        if git_hashes:
            print(f"  GET {backend_url}/tasks/all?git_hash={git_hashes[0]}&limit=50")
        if sim_suites:
            print(f"  GET {backend_url}/tasks/all?sim_suites={sim_suites[0]}&limit=50")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed eval_tasks data for manual testing.")
    parser.add_argument("--backend-url", default="http://localhost:8000", help="Base URL of the running backend")
    parser.add_argument("--user-email", default="seeder@example.com", help="Auth header for initial token creation")
    parser.add_argument("--policies", type=int, default=5, help="Number of policies to create")
    parser.add_argument("--tasks-per-policy", type=int, default=8, help="Tasks per policy")
    parser.add_argument(
        "--assignees",
        nargs="+",
        default=["worker_alpha", "worker_beta", "worker_gamma"],
        help="Assignee names to distribute claims",
    )
    parser.add_argument(
        "--sim-suites",
        nargs="+",
        default=["navigation", "memory", "arena", "wfc"],
        help="Simulation suite names to cycle",
    )
    parser.add_argument(
        "--git-hashes",
        nargs="+",
        default=[],
        help="Optional list of git hashes to cycle; random ones generated if empty",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    hashes = args.git_hashes or [_rand_commit_hex() for _ in range(max(20, args.policies * args.tasks_per_policy))]
    seed(
        backend_url=args.backend_url,
        user_email=args.user_email,
        num_policies=args.policies,
        tasks_per_policy=args.tasks_per_policy,
        assignees=args.assignees,
        sim_suites=args.sim_suites,
        git_hashes=hashes,
        seed_value=args.seed,
    )
