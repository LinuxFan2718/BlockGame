use std::io::{self, BufRead, Write};
use std::process::ExitCode;

use tic_tac_toe::{Agent, Game, Outcome, Player, RandomAgent};

const HELP: &str = "Tic-tac-toe: X always starts; enter a square 1-9, or q to quit.

Usage: cargo run -- [OPTIONS]
  --x human|random   X's controller (default: human)
  --o human|random   O's controller (default: random)
  --games N          Number of games, at least 1 (default: 1)
  --seed N           Reproducible bot seed, a u64 (default: 42)
  --quiet            Only print totals; requires two bots
  --help, -h         Show this help

Examples:
  cargo run
  cargo run -- --x human --o human
  cargo run -- --x random --o human
  cargo run --release -- --x random --o random --games 10000 --seed 42 --quiet";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Human,
    Random,
}

impl Kind {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "human" => Ok(Self::Human),
            "random" => Ok(Self::Random),
            _ => Err(format!("unknown controller {value:?}; use human or random")),
        }
    }
}

struct Config {
    x: Kind,
    o: Kind,
    games: u64,
    seed: u64,
    quiet: bool,
}

fn parse_args(args: impl Iterator<Item = String>) -> Result<Option<Config>, String> {
    let mut args = args;
    let mut config = Config {
        x: Kind::Human,
        o: Kind::Random,
        games: 1,
        seed: 42,
        quiet: false,
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--help" | "-h" => return Ok(None),
            "--quiet" => config.quiet = true,
            "--x" | "--o" | "--games" | "--seed" => {
                let value = args.next().ok_or_else(|| format!("{arg} needs a value"))?;
                match arg.as_str() {
                    "--x" => config.x = Kind::parse(&value)?,
                    "--o" => config.o = Kind::parse(&value)?,
                    "--games" => {
                        config.games = value
                            .parse()
                            .map_err(|_| "--games must be a positive u64".to_string())?;
                        if config.games == 0 {
                            return Err("--games must be at least 1".to_string());
                        }
                    }
                    "--seed" => {
                        config.seed = value
                            .parse()
                            .map_err(|_| "--seed must be a u64".to_string())?;
                    }
                    _ => unreachable!(),
                }
            }
            _ => return Err(format!("unknown option {arg:?}")),
        }
    }
    if config.quiet && (config.x == Kind::Human || config.o == Kind::Human) {
        return Err("--quiet requires --x random --o random".to_string());
    }
    Ok(Some(config))
}

enum Controller {
    Human,
    Random(RandomAgent),
}

impl Controller {
    fn new(kind: Kind, seed: u64) -> Self {
        match kind {
            Kind::Human => Self::Human,
            Kind::Random => Self::Random(RandomAgent::new(seed)),
        }
    }
}

fn human_action(
    player: Player,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> io::Result<Option<usize>> {
    let mut line = String::new();
    loop {
        write!(output, "{player}, choose a square (1-9, q to quit): ")?;
        output.flush()?;
        line.clear();
        if input.read_line(&mut line)? == 0 || line.trim().eq_ignore_ascii_case("q") {
            return Ok(None);
        }
        if let Ok(square @ 1..=9) = line.trim().parse::<usize>() {
            return Ok(Some(square - 1));
        }
        writeln!(output, "Please enter a number from 1 to 9, or q.")?;
    }
}

fn run(config: Config, input: &mut impl BufRead, output: &mut impl Write) -> io::Result<()> {
    let mut x = Controller::new(config.x, config.seed);
    let mut o = Controller::new(config.o, config.seed ^ 0xa076_1d64_78bd_642f);
    let (mut x_wins, mut o_wins, mut draws) = (0_u64, 0_u64, 0_u64);
    for number in 1..=config.games {
        let mut game = Game::new();
        if !config.quiet {
            writeln!(
                output,
                "\nGame {number}/{} (X starts)\n{game}\n",
                config.games
            )?;
        }
        while let Some(player) = game.current_player() {
            let controller = match player {
                Player::X => &mut x,
                Player::O => &mut o,
            };
            let action = match controller {
                Controller::Human => match human_action(player, input, output)? {
                    Some(action) => action,
                    None => {
                        writeln!(output, "\nStopped; the unfinished game is not counted.")?;
                        return writeln!(
                            output,
                            "Completed {} game(s): X wins {x_wins}, O wins {o_wins}, draws {draws}.",
                            x_wins + o_wins + draws
                        );
                    }
                },
                Controller::Random(bot) => {
                    let action = bot.select_action(&game.observe(player)).ok_or_else(|| {
                        io::Error::other("bot returned no action in a running game")
                    })?;
                    if !config.quiet {
                        writeln!(output, "{player} (random) chooses {}.", action + 1)?;
                    }
                    action
                }
            };
            match game.step(action) {
                Ok(_) => {
                    if !config.quiet {
                        writeln!(output, "\n{game}\n")?;
                    }
                }
                Err(error) => match controller {
                    Controller::Human => writeln!(output, "That square is occupied; try again.")?,
                    Controller::Random(_) => return Err(io::Error::other(error)),
                },
            }
        }
        if !config.quiet {
            writeln!(output, "{}.", game.outcome())?;
        }
        match game.outcome() {
            Outcome::Win(Player::X) => x_wins += 1,
            Outcome::Win(Player::O) => o_wins += 1,
            Outcome::Draw => draws += 1,
            Outcome::InProgress => unreachable!(),
        }
    }
    writeln!(
        output,
        "Completed {} game(s): X wins {x_wins}, O wins {o_wins}, draws {draws}. Seed: {}.",
        config.games, config.seed
    )
}

fn main() -> ExitCode {
    match parse_args(std::env::args().skip(1)) {
        Ok(None) => {
            println!("{HELP}");
            ExitCode::SUCCESS
        }
        Ok(Some(config)) => match run(config, &mut io::stdin().lock(), &mut io::stdout().lock()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("Error: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Error: {error}\nUse --help for usage.");
            ExitCode::from(2)
        }
    }
}
