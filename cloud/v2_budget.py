"""Credit guard for v2 Modal work: refuse launches that could exceed the cap.

The account has $130 of credits; real money must never be spent. A launch is
allowed only if current metered cost plus its worst-case cost (rate × timeout)
stays below CAP_USD, which leaves a margin for delayed metering.

    python cloud/v2_budget.py status
    python cloud/v2_budget.py check --gpu L4 --cpu 16 --memory-gib 32 --hours 6
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROFILE = "linuxfan2718"
CAP_USD = 118.0
MODAL = str(Path(__file__).resolve().parent.parent / ".venv-modal/bin/modal")


def cli(*args):
    return json.loads(
        subprocess.check_output([MODAL, *args, "--json", "--profile", PROFILE], text=True)
    )


def status():
    billing = cli("billing", "summary")
    apps = [a for a in cli("app", "list") if str(a.get("state", "")).lower() != "stopped"]
    return {
        "metered_usd": float(billing["metered_cost"]),
        "billed_usd": float(billing["billed_cost"]),
        "cap_usd": CAP_USD,
        "headroom_usd": round(CAP_USD - float(billing["metered_cost"]), 2),
        "running_apps": [(a.get("app_id"), a.get("description"), a.get("tasks")) for a in apps],
    }


def hourly(gpu, cpu, memory_gib):
    rates = cli("billing", "rates")
    gpu_rate = float(rates[f"gpu_hour_cost_{gpu.lower()}"]) if gpu else 0.0
    return gpu_rate + cpu * float(rates["cpu_hour_cost"]) + memory_gib * float(rates["mem_gib_hour_cost"])


def check(gpu, cpu, memory_gib, hours):
    s = status()
    worst = hourly(gpu, cpu, memory_gib) * hours
    allowed = s["metered_usd"] + worst <= CAP_USD and s["billed_usd"] == 0
    return {**s, "hourly_usd": round(worst / hours, 4), "worst_case_usd": round(worst, 2), "allowed": allowed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "check"))
    parser.add_argument("--gpu", default="")
    parser.add_argument("--cpu", type=float, default=0)
    parser.add_argument("--memory-gib", type=float, default=0)
    parser.add_argument("--hours", type=float, default=1)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2))
        return
    result = check(args.gpu, args.cpu, args.memory_gib, args.hours)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["allowed"] else 1)


if __name__ == "__main__":
    main()
