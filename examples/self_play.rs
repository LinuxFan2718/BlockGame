//! Record a complete episode without terminal input or changes to the rules.
use tic_tac_toe::{Agent, Game, Observation, Player, RandomAgent};

struct Turn {
    player: Player,
    observation: Observation,
    action: usize,
}

fn main() {
    let mut game = Game::new();
    let mut x = RandomAgent::new(42);
    let mut o = RandomAgent::new(123);
    let mut trajectory = Vec::new();

    while let Some(player) = game.current_player() {
        let observation = game.observe(player);
        let bot = match player {
            Player::X => &mut x,
            Player::O => &mut o,
        };
        let action = bot.select_action(&observation).unwrap();
        trajectory.push(Turn {
            player,
            observation,
            action,
        });
        game.step(action).unwrap();
    }

    println!("{game}\n{}", game.outcome());
    for turn in trajectory {
        // With no intermediate rewards and gamma = 1, every turn's return is
        // its player's final reward. This records data; it does not train a bot.
        let episode_return = game.outcome().reward_for(turn.player);
        println!(
            "{}: state {:?}, action {}, return {episode_return:+}",
            turn.player, turn.observation.cells, turn.action
        );
    }
}
