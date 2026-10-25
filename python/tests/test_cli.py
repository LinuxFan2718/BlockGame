import pytest
from blokus_rl.cli import main
from blokus_rl.learning import TrainConfig, configure_torch, save_checkpoint
from blokus_rl.model import ActorCritic


def test_inspect_shows_masked_probabilities_and_gradients(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["blokus-rl", "inspect"])
    main()
    output = capsys.readouterr().out
    assert "5,450" in output
    assert "square 1: 0.0000" in output
    assert "shared trunk gradient norm" in output
    assert "not PPO training" in output


@pytest.mark.parametrize("human", ["X", "O"])
def test_human_can_quit_playing_a_checkpoint(human, tmp_path, monkeypatch, capsys):
    configure_torch(42)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, ActorCritic(), TrainConfig())
    monkeypatch.setattr("sys.argv", ["blokus-rl", "play", str(checkpoint), "--human", human])
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    main()
    assert f"You are {human}" in capsys.readouterr().out


def test_invalid_training_options_fail_before_creating_output(tmp_path, monkeypatch):
    out = tmp_path / "invalid"
    monkeypatch.setattr("sys.argv", ["blokus-rl", "train", "--updates", "0", "--out", str(out)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not out.exists()
