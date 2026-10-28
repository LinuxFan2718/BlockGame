import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch
from blokus_rl import duo_web
from blokus_rl._native import DuoGame
from blokus_rl.duo_conv import DuoConvActorCritic
from blokus_rl.duo_device import inference_device
from blokus_rl.duo_learning import choose
from blokus_rl.duo_mcts import search_games
from blokus_rl.learning import configure_torch


class StubModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def wait_for_bot(session):
    deadline = time.monotonic() + 5
    while session.state()["status"] == "thinking":
        assert time.monotonic() < deadline, "bot did not finish"
        time.sleep(0.002)
    state = session.state()
    assert state["status"] != "error", state["error"]
    return state


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(duo_web, "choose", lambda game, *_: game.legal_actions()[0])
    game = duo_web.LocalGame({"test": ("Test", "unused")}, "cpu", policy_loader=lambda _: StubModel())
    game.new({"human": 0, "simulations": 0, "device": "cpu"})
    yield game
    game.close()


def test_device_selection_and_explicit_errors(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert str(inference_device()) == "mps"
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert str(inference_device()) == "cpu"
    with pytest.raises(ValueError, match="Metal is unavailable"):
        inference_device("mps")
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        inference_device("cuda")


def test_illegal_and_stale_moves_cannot_change_board(session):
    before = session.state()
    with pytest.raises(ValueError, match="not legal"):
        session.move({**before, "action": -1})
    with pytest.raises(duo_web.StaleGame):
        session.move({**before, "version": -1, "action": before["legal_moves"][0]["id"]})
    assert session.state()["board"] == before["board"]


def test_undo_reverts_human_and_bot_and_invalidates_old_requests(session):
    before = session.state()
    session.move({**before, "action": before["legal_moves"][0]["id"]})
    after = wait_for_bot(session)
    assert len(after["actions"]) == 2
    restored = session.undo(after)
    assert restored["board"] == before["board"]
    assert restored["version"] > after["version"]
    assert restored["status"] == "your_turn"
    with pytest.raises(duo_web.StaleGame):
        session.move({**before, "action": before["legal_moves"][0]["id"]})


@pytest.mark.parametrize("operation", ["new", "undo"])
def test_inflight_bot_cannot_write_into_reset_or_undone_game(session, monkeypatch, operation):
    entered, release = threading.Event(), threading.Event()

    def blocked(game, *_):
        entered.set()
        assert release.wait(5)
        return game.legal_actions()[0]

    monkeypatch.setattr(duo_web, "choose", blocked)
    before = session.state()
    pending = session.move({**before, "action": before["legal_moves"][0]["id"]})
    assert entered.wait(5)
    try:
        restored = (session.new({"human": 0, "simulations": 0, "device": "cpu"})
                    if operation == "new" else session.undo(pending))
    finally:
        release.set()
    # Drain the single executor so the stale completion has definitely run.
    session.executor.submit(lambda: None).result(timeout=5)
    assert session.state()["board"] == restored["board"] == [0] * 196
    assert session.state()["status"] == "your_turn"


@pytest.mark.parametrize("human", [0, 1])
def test_complete_game_passes_and_scores_match_rust(session, human):
    state = session.new({"human": human, "simulations": 0, "device": "cpu"})
    for _ in range(100):
        state = wait_for_bot(session)
        if state["status"] == "finished":
            break
        assert state["current_player"] == human
        move = state["legal_moves"][0]
        state = session.move({**state, "action": move["id"]})
    else:
        pytest.fail("game did not terminate")
    replay = DuoGame()
    for row in state["actions"]:
        assert row["player"] == replay.current_player
        assert row["action"] in replay.legal_actions()
        replay.step(row["action"])
    assert replay.terminated
    assert replay.scores == state["scores"]
    assert list(replay.board) == state["board"]
    assert state["winner"] in ("human", "bot", "draw")
    # End-of-game automatic passes must not become the "human move" being undone.
    last_placement = max(i for i, row in enumerate(state["actions"])
                         if row["player"] == human and row["action"] != duo_web.PASS)
    restored = session.undo(state)
    assert restored["actions"] == state["actions"][:last_placement]
    assert restored["status"] == "your_turn"


def test_http_contract_rejects_foreign_origins_and_stale_posts(session):
    server = duo_web.PlayServer(0, session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def get(path):
        with urlopen(base + path) as response:
            return json.load(response)

    def post(data, **headers):
        return urlopen(Request(base + "/api/move", data=json.dumps(data).encode(), headers=headers))

    try:
        config = get("/api/config")
        state = get("/api/state")
        assert config["application"] == "blokus-duo-local"
        assert len(config["pieces"]) == 21
        with pytest.raises(HTTPError) as error:
            post({**state, "action": 0})
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post(state, **{"X-Blokus-Token": config["token"], "Origin": "https://example.com"})
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            post({"id": state["id"], "version": -1}, **{"X-Blokus-Token": config["token"]})
        assert error.value.code == 409
        assert get("/api/state")["board"] == state["board"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_raw_and_search_on_inference_device(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal GPU access unavailable in this process")
    configure_torch(42)
    model = DuoConvActorCritic().to(device).eval()
    game = DuoGame()
    before = game.render()
    assert choose(game, model, np.random.default_rng(42)) in game.legal_actions()
    result = search_games(model, [game], 16)[0]
    assert result.action() in game.legal_actions()
    assert result.visits.sum() == 16
    assert np.isfinite(result.priors).all()
    assert game.render() == before
