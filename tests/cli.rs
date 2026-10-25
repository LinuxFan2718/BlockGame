use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], input: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tic-tac-toe"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn humans_can_complete_a_game_and_recover_from_bad_input() {
    let result = run(
        &["--x", "human", "--o", "human"],
        "hello\n0\n10\n1\n1\n4\n2\n5\n3\n",
    );
    assert!(result.status.success());
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.contains("Please enter a number from 1 to 9"));
    assert!(stdout.contains("That square is occupied"));
    assert!(stdout.contains("X wins 1, O wins 0, draws 0"));
}

#[test]
fn quitting_and_end_of_input_do_not_count_an_unfinished_game() {
    for input in ["q\n", ""] {
        let result = run(&[], input);
        assert!(result.status.success());
        assert!(
            String::from_utf8(result.stdout)
                .unwrap()
                .contains("Completed 0 game(s)")
        );
    }
}

#[test]
fn bot_batches_are_repeatable_and_need_no_input() {
    let args = [
        "--x", "random", "--o", "random", "--games", "100", "--seed", "0", "--quiet",
    ];
    let first = run(&args, "");
    let second = run(&args, "");
    assert!(first.status.success());
    assert!(second.status.success());
    assert_eq!(first.stdout, second.stdout);
    let stdout = String::from_utf8(first.stdout).unwrap();
    assert!(stdout.starts_with("Completed 100 game(s):"));
    assert_eq!(stdout.lines().count(), 1);
}

#[test]
fn invalid_options_fail_and_help_succeeds() {
    for args in [
        vec!["--unknown"],
        vec!["--x"],
        vec!["--o", "clever"],
        vec!["--games", "0"],
        vec!["--games", "-1"],
        vec!["--seed", "nope"],
        vec!["--quiet"],
    ] {
        let result = run(&args, "");
        assert_eq!(result.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8(result.stderr).unwrap().contains("Error:"));
    }
    let result = run(&["--help"], "");
    assert!(result.status.success());
    assert!(String::from_utf8(result.stdout).unwrap().contains("Usage:"));
}
