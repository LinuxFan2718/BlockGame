import time

import pytest

from blokus_rl.v2 import ladder

GB = 1e9


def test_pentobi_memory_matches_pentobi_sizing():
    sixteen = 16 * 2**30
    assert ladder.pentobi_bytes(9, sixteen) == 2_000_000_000 + ladder.PENTOBI_OVERHEAD
    assert 0.5 * GB < ladder.pentobi_bytes(8, sixteen) < 0.65 * GB
    assert ladder.pentobi_bytes(1, sixteen) == ladder.pentobi_bytes(5, sixteen)
    # Never more than a quarter of physical memory.
    assert ladder.pentobi_bytes(9, 4 * 2**30) == 2**30 + ladder.PENTOBI_OVERHEAD


def test_concurrent_games_fit_the_budget():
    each = ladder.pentobi_bytes(9)
    assert ladder.concurrent_games(9, 64, 100, 3.5 * each) == 3
    assert ladder.concurrent_games(9, 64, 1, 100 * each) == 2
    with pytest.raises(MemoryError):
        ladder.concurrent_games(9, 64, 100, 0.5 * each)


def test_memory_guard_kills_pentobi_when_memory_runs_short():
    class Opponent:
        killed = False

        def pids(self):
            return []

        def kill(self):
            self.killed = True

    opponent = Opponent()
    guard = ladder.MemoryGuard(opponent, floor=1e15, progress=lambda: "0/0", interval=0.01)
    deadline = time.monotonic() + 5
    while not opponent.killed and time.monotonic() < deadline:
        time.sleep(0.01)
    guard.stop()
    assert opponent.killed and guard.tripped
    assert ladder.available_memory() > 0
