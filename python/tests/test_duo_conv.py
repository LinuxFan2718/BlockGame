import numpy as np
import torch
from blokus_rl._native import DuoGame
from blokus_rl.duo_conv import DuoConvActorCritic
from blokus_rl.duo_env import CELLS, PASS, decision
from blokus_rl.duo_learning import (
    DuoConfig,
    collect_duo,
    load_duo,
    make_duo_loss,
    optimize_duo,
    pack,
    save_duo,
)
from blokus_rl.learning import configure_torch


def test_spatial_action_mapping_matches_native_geometry_in_both_perspectives():
    configure_torch(27)
    game, model = DuoGame(), DuoConvActorCritic(16)
    lookup = {}
    for action in range(PASS):
        try:
            _, _, _, cells = game.describe(action)
        except ValueError:
            continue
        key = frozenset(cells)
        assert key not in lookup
        lookup[key] = action
    rng = np.random.default_rng(82)
    seen_players = set()
    while not game.terminated:
        player = game.current_player
        seen_players.add(player)
        _, actions, candidates = decision(game)
        indices = model.candidate_indices(torch.from_numpy(candidates)).tolist()
        for action, actual in zip(actions, indices, strict=True):
            if action == PASS:
                assert actual == PASS
                continue
            _, _, _, absolute_cells = game.describe(int(action))
            relative_cells = (CELLS - 1 - c for c in absolute_cells) if player else absolute_cells
            assert actual == lookup[frozenset(relative_cells)]
        # Reordering a shape's cells cannot change its anchor/orientation score.
        reversed_cells = candidates.copy()
        reversed_cells[:, 1:] = reversed_cells[:, 1:][:, ::-1]
        assert torch.equal(
            model.candidate_indices(torch.from_numpy(candidates)),
            model.candidate_indices(torch.from_numpy(reversed_cells)),
        )
        game.step(int(rng.choice(actions)))
    assert seen_players == {0, 1}


def test_spatial_ppo_and_checkpoint_roundtrip(tmp_path):
    configure_torch(42)
    config = DuoConfig(
        updates=1, episodes_per_update=2, epochs=1, hidden_size=16, architecture="conv"
    )
    model = DuoConvActorCritic(16)
    before = {k: v.clone() for k, v in model.named_parameters()}
    records, _ = collect_duo(model, config, np.random.default_rng(1), [])
    batch = pack(records[:5])
    with torch.no_grad():
        distribution, _ = model.distribution(
            batch["observation"], batch["candidates"], batch["action_mask"]
        )
        assert torch.allclose(
            distribution.log_prob(batch["action"]), batch["action_log_prob"], atol=2e-6
        )
        assert torch.count_nonzero(distribution.probs[~batch["action_mask"]]) == 0
    stats = optimize_duo(
        model,
        make_duo_loss(model, config),
        torch.optim.Adam(model.parameters(), lr=0.001),
        records,
        config,
    )
    assert all(np.isfinite(x) for x in stats.values())
    assert any(not torch.equal(v, before[k]) for k, v in model.named_parameters())
    path = tmp_path / "conv.pt"
    save_duo(path, model, config, 1)
    loaded = load_duo(path)
    assert loaded.architecture == "conv"
    with torch.no_grad():
        assert torch.equal(
            model.logits(batch["observation"], batch["candidates"]),
            loaded.logits(batch["observation"], batch["candidates"]),
        )
