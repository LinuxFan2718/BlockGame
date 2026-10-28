"""Compare isolated Rust move-generation/PUCT prototypes on saved traces.

Copies source into the output directory; never changes the production engine,
checkpoints, training state or opponent results. Uses only the standard library.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

CORNER_METHOD = """
    pub fn legal_actions(&self) -> Vec<usize> {
        if self.terminal { return vec![]; }
        let mut anchors = Bits::default();
        if self.last_piece[self.turn].is_some() {
            for i in 0..4 {
                anchors.0[i] = self.corners[self.turn].0[i]
                    & !(self.edges[self.turn].0[i] | self.occupied.0[i]);
            }
        } else {
            for cell in self.required_starts() { anchors.insert(cell); }
        }
        let mut seen = [0_u64; PASS.div_ceil(64)];
        let mut actions = Vec::new();
        for word in 0..4 {
            let mut bits = anchors.0[word];
            while bits != 0 {
                let cell = word * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                for &action in &catalog().by_cell[cell] {
                    let bit = 1_u64 << (action % 64);
                    if seen[action / 64] & bit != 0 { continue; }
                    seen[action / 64] |= bit;
                    let placement = catalog().placements[action].as_ref().unwrap();
                    if self.legal_placement(placement) { actions.push(action); }
                }
            }
        }
        actions.sort_unstable(); // Preserve the canonical ordering and tie breaks.
        if actions.is_empty() { actions.push(PASS); }
        actions
    }
"""

DRIVER = r"""
#![allow(dead_code, unused_variables)]
mod duo;
mod duo_search;
use std::hint::black_box;
use std::time::Instant;

fn mix(hash: &mut u64, value: u64) {
    *hash = (*hash ^ value).wrapping_mul(1099511628211);
}

fn main() {
    let mut states = Vec::new();
    let mut roots = Vec::new();
    let mut passes = 0;
    for (game_index, line) in include_str!("actions.txt").lines().enumerate() {
        let mut game = duo::Game::new();
        for (ply, action) in line.split_whitespace().map(|s|s.parse::<usize>().unwrap()).enumerate() {
            assert_eq!(game.legal_actions(), game.legal_actions_scan());
            if action == duo::PASS { passes += 1; }
            if game_index < 3 && [0, 8, 16, 24].contains(&ply) { roots.push(game.clone()); }
            states.push(game.clone());
            game.step(action).unwrap();
        }
        assert!(game.is_terminal());
        assert_eq!(game.legal_actions(), game.legal_actions_scan());
        states.push(game);
    }
    let mut digest = 14695981039346656037_u64;
    let started = Instant::now();
    for game in &states {
        let actions = black_box(game).legal_actions();
        for action in actions { mix(&mut digest, action as u64); }
    }
    let legal_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut search_digest = 14695981039346656037_u64;
    let mut simulations = 0;
    #[cfg(probe_verify)]
    let mut signatures = Vec::new();
    for game in &roots {
        let mut tree = duo_search::Search::new(game.clone(), 1.5, 1.0).unwrap();
        while tree.completed() < 1024 {
            let leaves = tree.request((1024 - tree.completed()).min(8)).unwrap();
            for (token, leaf) in leaves {
                #[cfg(probe_verify)]
                assert_eq!(tree.pending_actions(token).unwrap(), leaf.legal_actions_scan());
                let count = tree.pending_actions(token).unwrap().len();
                tree.expand(token, vec![1.0; count], 0.0).unwrap();
            }
        }
        let stats = tree.stats().unwrap();
        assert_eq!(stats.simulations, 1024);
        assert_eq!(stats.visits.iter().sum::<u32>(), 1024);
        simulations += stats.simulations;
        mix(&mut search_digest, stats.nodes as u64);
        for i in 0..stats.actions.len() {
            mix(&mut search_digest, stats.actions[i] as u64);
            mix(&mut search_digest, stats.visits[i] as u64);
            mix(&mut search_digest, stats.priors[i].to_bits() as u64);
            mix(&mut search_digest, stats.means[i].to_bits() as u64);
        }
        #[cfg(probe_verify)]
        signatures.push(format!("{stats:?}"));
    }
    let search_seconds = started.elapsed().as_secs_f64();
    #[cfg(probe_verify)]
    std::fs::write(std::env::current_exe().unwrap().parent().unwrap().join("root-stats.txt"),
                  signatures.join("\n")).unwrap();
    println!("{{\"positions\":{},\"forced_pass_moves\":{},\"legal_digest\":{},\"search_digest\":{},\"simulations\":{},\"legal_seconds\":{},\"search_seconds\":{}}}",
        states.len(), passes, digest, search_digest, simulations, legal_seconds, search_seconds);
}
"""


def corner_source(source):
    """Add a per-cell placement index and keep the original scan as an oracle."""
    source = source.replace(
        "    placements: Vec<Option<Placement>>,",
        "    placements: Vec<Option<Placement>>,\n    by_cell: Vec<Vec<usize>>,",
    )
    needle = """        Catalog {
            pieces,
            orientations,
            placements,
        }"""
    replacement = """        let mut by_cell = vec![Vec::new(); CELLS];
        for p in placements.iter().flatten() {
            for &cell in &p.cells { by_cell[cell].push(p.action); }
        }
        Catalog {
            pieces,
            orientations,
            placements,
            by_cell,
        }"""
    assert needle in source
    source = source.replace(needle, replacement)
    start = source.index("    pub fn legal_actions(&self)")
    end = source.index("    /// The score", start)
    original = source[start:end].replace("fn legal_actions(", "fn legal_actions_scan(")
    return source[:start] + original + CORNER_METHOD + "\n" + source[end:]


def priority_source(source):
    needle = """                let mut best = 0;
                for i in 1..node.actions.len() {
                    if priority(i) > priority(best) {
                        best = i;
                    }
                }"""
    replacement = """                let mut best = 0;
                let mut best_priority = priority(0);
                for i in 1..node.actions.len() {
                    let candidate_priority = priority(i);
                    if candidate_priority > best_priority {
                        best = i;
                        best_priority = candidate_priority;
                    }
                }"""
    assert needle in source
    return source.replace(needle, replacement)


def verify_sources(out, variants, rustc):
    """Outside timing: compare every pending leaf with the scan and full stats."""
    outputs = {}
    for variant in variants:
        directory = out / variant
        driver = directory / "verify-main.rs"
        driver.write_text(DRIVER)
        subprocess.run(
            [
                rustc,
                "--edition=2024",
                "-O",
                "--cfg",
                "probe_verify",
                str(driver),
                "-o",
                str(directory / "verify-probe"),
            ],
            check=True,
        )
        outputs[variant] = json.loads(subprocess.check_output([str(directory / "verify-probe")]))
    reference = (out / "baseline/root-stats.txt").read_bytes()
    assert all((out / variant / "root-stats.txt").read_bytes() == reference for variant in variants)
    report = {
        "every_fixture_legal_list_exact": True,
        "every_requested_search_leaf_matches_scan": True,
        "all_full_root_stats_exact": True,
        "root_stats_sha256": hashlib.sha256(reference).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": outputs,
    }
    (out / "correctness.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    variants = ("baseline", "corner-index", "priority", "corner-priority")
    rustc = str(Path.home() / ".cargo/bin/rustc")
    if args.verify_existing:
        print(json.dumps(verify_sources(args.out, variants, rustc), indent=2))
        return
    args.out.mkdir(exist_ok=False, parents=True)
    root = Path(__file__).resolve().parent.parent
    source = (root / "src/duo.rs").read_text()
    search = (root / "src/duo_search.rs").read_text()
    pairs = sorted(args.pairs.glob("pair-*.json"))
    assert len(pairs) == 100
    fixtures, hashes = [], {}
    for pair in pairs:
        hashes[str(pair)] = hashlib.sha256(pair.read_bytes()).hexdigest()
        for game in json.loads(pair.read_text())["games"]:
            fixtures.append(" ".join(map(str, game["actions"])))
    for variant in variants:
        directory = args.out / variant
        directory.mkdir()
        engine = (
            corner_source(source)
            if variant.startswith("corner")
            else source.replace(
                "    pub fn legal_actions(&self)",
                "    pub fn legal_actions_scan(&self) -> Vec<usize> { self.legal_actions() }\n"
                "    pub fn legal_actions(&self)",
            )
        )
        (directory / "duo.rs").write_text(engine)
        (directory / "duo_search.rs").write_text(
            priority_source(search) if "priority" in variant else search
        )
        (directory / "main.rs").write_text(DRIVER)
        (directory / "actions.txt").write_text("\n".join(fixtures) + "\n")
        subprocess.run(
            [
                rustc,
                "--edition=2024",
                "-O",
                str(directory / "main.rs"),
                "-o",
                str(directory / "probe"),
            ],
            check=True,
        )
    results = {variant: [] for variant in variants}
    for repeat in range(7):
        order = variants if repeat % 2 == 0 else tuple(reversed(variants))
        for variant in order:
            result = json.loads(subprocess.check_output([str(args.out / variant / "probe")]))
            reference = next(iter(results["baseline"]), result)
            for key in (
                "positions",
                "forced_pass_moves",
                "legal_digest",
                "search_digest",
                "simulations",
            ):
                assert result[key] == reference[key], (variant, key)
            results[variant].append(result)
        print(json.dumps({"repeat": repeat + 1, "complete": True}), flush=True)
    report = {
        "variants": {
            name: {
                "runs": runs,
                "median_legal_seconds": statistics.median(r["legal_seconds"] for r in runs),
                "median_search_seconds": statistics.median(r["search_seconds"] for r in runs),
            }
            for name, runs in results.items()
        },
        "fixture_sha256": hashes,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("src/duo.rs", "src/duo_search.rs")
        },
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rustc": subprocess.check_output([rustc, "--version"], text=True).strip(),
        "games_replayed": len(fixtures),
        "gradient_updates": 0,
        "limitation": "Local native engine only, uniform priors/value0; not neural self-play or GPU throughput.",
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    verify_sources(args.out, variants, rustc)
    print(
        json.dumps(
            {k: {a: b for a, b in v.items() if a != "runs"} for k, v in report["variants"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
