"""Exercise paid-launch guards offline: no Modal SDK/network/GPU work is invoked."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    class App:
        app_id = "own-app"

        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            assert kwargs["timeout"] == 14400 and kwargs["retries"] == 0
            assert kwargs["max_containers"] == 1
            return lambda fn: SimpleNamespace(
                spawn=Mock(return_value=SimpleNamespace(object_id="fc-test"))
            )

        def local_entrypoint(self):
            return lambda fn: fn

    fake = SimpleNamespace(
        is_local=lambda: False,
        App=App,
        Volume=SimpleNamespace(from_name=lambda *a, **k: None),
    )
    monkeypatch.setitem(sys.modules, "modal", fake)
    spec = importlib.util.spec_from_file_location(
        "replication_offline", ROOT / "cloud/modal_control_replication.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "runs").mkdir()
    (tmp_path / "python/blokus_rl").mkdir(parents=True)
    for name in ("duo_conv", "duo_residual", "duo_learning", "learning", "duo_crossplay"):
        (tmp_path / f"python/blokus_rl/{name}.py").write_text("# fixture\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    (tmp_path / "runs/integrated-validation-verified.json").write_text(
        json.dumps(
            {"passed": True, "payload_sha256": {}, "selected_runtime": {"bulk_inference": True}}
        )
    )
    (tmp_path / "runs/control-replication-preserved-before.json").write_text("{}")
    helper, protocol = tmp_path / "helper.py", tmp_path / "protocol.json"
    helper.write_text("# helper fixture\n")
    protocol.write_text((ROOT / "docs/control-replication-protocol.json").read_text())
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "HELPER", helper)
    monkeypatch.setattr(module, "PROTOCOL", protocol)
    monkeypatch.setattr(module, "MANIFEST", manifest)
    state = {
        "apps": [{"app_id": "own-app", "state": "ephemeral", "tasks": "0"}],
        "billing": {"metered_cost": 53.03137374},
        "rates": {"gpu_hour_cost_l40s": 1.95, "cpu_hour_cost": 0.0473, "mem_gib_hour_cost": 0.008},
    }

    def output(command, **kwargs):
        if command[0] == "git":
            return "test-commit\n"
        key = "apps" if "app" in command else "rates" if "rates" in command else "billing"
        return json.dumps(state[key])

    monkeypatch.setattr(module.subprocess, "check_output", output)
    return module, state


def test_one_shot_receipt_and_reserve(launcher):
    module, _ = launcher
    module.main()
    module.study.spawn.assert_called_once()
    execution = module.study.spawn.call_args.args[0]
    assert execution["reserved_usd"] == pytest.approx(10.29461)
    assert execution["account_commitment_ceiling_usd"] == pytest.approx(64.29461)
    assert execution["config"]["full_simulations"] == 1024
    assert execution["config"]["seed"] == 1071
    with pytest.raises(ValueError, match="duplicate"):
        module.main()
    module.study.spawn.assert_called_once()


@pytest.mark.parametrize("reason", ["active", "credits", "rates"])
def test_no_submission_if_guards_fail(launcher, reason):
    module, state = launcher
    if reason == "active":
        state["apps"].append({"app_id": "other", "state": "detached", "tasks": "1"})
    elif reason == "credits":
        state["billing"]["metered_cost"] = 71
    else:
        state["rates"]["gpu_hour_cost_l40s"] = 2
    with pytest.raises(ValueError):
        module.main()
    module.study.spawn.assert_not_called()
    assert not (module.ROOT / "runs/modal-control-replication-cost-ledger.jsonl").exists()
