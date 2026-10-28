//! On-demand PUCT search over live Blokus Duo states.
//!
//! The caller supplies neural priors and values for requested leaves. There is
//! no position database, heuristic opponent or evaluation oracle in this module.
//! Pending paths carry virtual loss so several leaves can be evaluated together.
use crate::duo::Game;

#[derive(Clone, Debug)]
struct Node {
    game: Game,
    player: usize,
    actions: Vec<usize>,
    priors: Vec<f32>,
    visits: Vec<u32>,
    virtual_visits: Vec<u32>,
    sums: Vec<f32>,
    children: Vec<Option<usize>>,
    expanded: bool,
    pending: Option<Vec<(usize, usize)>>,
}

impl Node {
    fn new(game: Game, player: usize) -> Self {
        Self {
            game,
            player,
            actions: vec![],
            priors: vec![],
            visits: vec![],
            virtual_visits: vec![],
            sums: vec![],
            children: vec![],
            expanded: false,
            pending: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RootStats {
    pub actions: Vec<usize>,
    pub visits: Vec<u32>,
    pub priors: Vec<f32>,
    /// Every Q is in the root player's perspective.
    pub means: Vec<f32>,
    pub simulations: usize,
    pub nodes: usize,
}

#[derive(Debug)]
pub struct Search {
    nodes: Vec<Node>,
    exploration: f32,
    virtual_loss: f32,
    completed: usize,
    pending_count: usize,
}

impl Search {
    pub fn new(game: Game, exploration: f32, virtual_loss: f32) -> Result<Self, String> {
        let player = game
            .current_player()
            .ok_or("cannot search a terminal game")?;
        if !exploration.is_finite()
            || exploration < 0.0
            || !virtual_loss.is_finite()
            || virtual_loss < 0.0
        {
            return Err("invalid exploration/virtual-loss setting".into());
        }
        Ok(Self {
            nodes: vec![Node::new(game, player)],
            exploration,
            virtual_loss,
            completed: 0,
            pending_count: 0,
        })
    }

    pub fn completed(&self) -> usize {
        self.completed
    }
    pub fn pending(&self) -> usize {
        self.pending_count
    }

    /// Legal order already generated for a pending leaf; avoid generating it
    /// again when encoding a neural observation across a language boundary.
    pub fn pending_actions(&self, token: usize) -> Result<&[usize], String> {
        let node = self.nodes.get(token).ok_or("unknown leaf token")?;
        if node.pending.is_none() {
            return Err("leaf token is not pending".into());
        }
        Ok(&node.actions)
    }

    fn backup(&mut self, path: &[(usize, usize)], mut value: f32, virtual_path: bool) {
        for &(node, action) in path.iter().rev() {
            value = -value; // Leaf/current player -> previous player, once per ply.
            let parent = &mut self.nodes[node];
            if virtual_path {
                parent.virtual_visits[action] -= 1;
            }
            parent.visits[action] += 1;
            parent.sums[action] += value;
        }
        // Expanding the initial root is setup, not a simulated root action.
        if !path.is_empty() {
            self.completed += 1;
        }
    }

    /// Request up to `budget` simulations. Terminal leaves are backed up here;
    /// nonterminal leaves are returned for a batched neural evaluation.
    /// Resolve all returned tokens before requesting the next wave.
    pub fn request(&mut self, budget: usize) -> Result<Vec<(usize, Game)>, String> {
        if budget == 0 {
            return Err("request budget must be positive".into());
        }
        if self.pending_count != 0 {
            return Err("resolve pending leaves first".into());
        }
        let mut leaves = Vec::new();
        for _ in 0..budget {
            let mut current = 0;
            let mut path = Vec::new();
            loop {
                let node = &self.nodes[current];
                if node.game.is_terminal() {
                    let scores = node.game.scores();
                    let value = (scores[node.player] - scores[1 - node.player]).signum() as f32;
                    self.backup(&path, value, false);
                    break;
                }
                if !node.expanded {
                    // A collision means this wave is already evaluating that
                    // leaf. Finish the wave; no duplicate inference/backups.
                    if node.pending.is_some() {
                        return Ok(leaves);
                    }
                    for &(parent, action) in &path {
                        self.nodes[parent].virtual_visits[action] += 1;
                    }
                    let node = &mut self.nodes[current];
                    node.actions = node.game.legal_actions();
                    node.pending = Some(path);
                    self.pending_count += 1;
                    leaves.push((current, node.game.clone()));
                    if current == 0 {
                        return Ok(leaves);
                    }
                    break;
                }
                let total: u32 = node
                    .visits
                    .iter()
                    .zip(&node.virtual_visits)
                    .map(|(real, virtual_)| real + virtual_)
                    .sum();
                let scale = ((total + 1) as f32).sqrt();
                let priority = |i: usize| {
                    let count = node.visits[i] + node.virtual_visits[i];
                    let q = if count == 0 {
                        0.0
                    } else {
                        (node.sums[i] - self.virtual_loss * node.virtual_visits[i] as f32)
                            / count as f32
                    };
                    q + self.exploration * node.priors[i] * scale / (1.0 + count as f32)
                };
                let mut best = 0;
                let mut best_priority = priority(0);
                for i in 1..node.actions.len() {
                    let candidate_priority = priority(i);
                    if candidate_priority > best_priority {
                        best = i;
                        best_priority = candidate_priority;
                    }
                }
                let child = if let Some(child) = node.children[best] {
                    child
                } else {
                    let mut game = node.game.clone();
                    game.step(node.actions[best])
                        .expect("stored legal search action");
                    let player = 1 - node.player;
                    let child = self.nodes.len();
                    self.nodes.push(Node::new(game, player));
                    self.nodes[current].children[best] = Some(child);
                    child
                };
                path.push((current, best));
                current = child;
            }
        }
        Ok(leaves)
    }

    /// Priors correspond to the legal-action order returned by the Rust game.
    /// Validation happens before consuming the pending token.
    pub fn expand(&mut self, token: usize, priors: Vec<f32>, value: f32) -> Result<(), String> {
        let node = self.nodes.get(token).ok_or("unknown leaf token")?;
        if node.pending.is_none() {
            return Err("leaf token is not pending".into());
        }
        if priors.len() != node.actions.len()
            || priors.is_empty()
            || priors.iter().any(|p| !p.is_finite() || *p < 0.0)
            || !value.is_finite()
            || !(-1.0..=1.0).contains(&value)
        {
            return Err("invalid neural priors or value".into());
        }
        let sum: f32 = priors.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("priors must have positive finite mass".into());
        }
        let node = &mut self.nodes[token];
        let path = node.pending.take().unwrap();
        node.priors = priors.into_iter().map(|p| p / sum).collect();
        let n = node.actions.len();
        node.visits = vec![0; n];
        node.virtual_visits = vec![0; n];
        node.sums = vec![0.0; n];
        node.children = vec![None; n];
        node.expanded = true;
        self.pending_count -= 1;
        self.backup(&path, value, true);
        Ok(())
    }

    pub fn stats(&self) -> Result<RootStats, String> {
        if self.pending_count != 0 {
            return Err("search has pending neural evaluations".into());
        }
        let node = &self.nodes[0];
        if !node.expanded {
            return Err("root has not been evaluated".into());
        }
        Ok(RootStats {
            actions: node.actions.clone(),
            visits: node.visits.clone(),
            priors: node.priors.clone(),
            means: node
                .sums
                .iter()
                .zip(&node.visits)
                .map(|(sum, count)| {
                    if *count == 0 {
                        0.0
                    } else {
                        sum / *count as f32
                    }
                })
                .collect(),
            simulations: self.completed,
            nodes: self.nodes.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pending_validation_and_exact_simulation_accounting() {
        let mut tree = Search::new(Game::new(), 1.5, 1.0).unwrap();
        assert!(tree.stats().is_err());
        let leaves = tree.request(8).unwrap();
        assert_eq!(leaves.len(), 1);
        assert!(tree.request(8).is_err());
        let n = leaves[0].1.legal_actions().len();
        assert!(tree.expand(0, vec![0.0; n], 0.0).is_err());
        assert_eq!(tree.pending(), 1);
        tree.expand(0, vec![1.0; n], 0.0).unwrap();
        assert_eq!(tree.completed(), 0);
        assert!(tree.expand(0, vec![1.0; n], 0.0).is_err());
        while tree.completed() < 64 {
            let leaves = tree.request((64 - tree.completed()).min(8)).unwrap();
            for (token, game) in leaves {
                tree.expand(token, vec![1.0; game.legal_actions().len()], 0.0)
                    .unwrap();
            }
        }
        let stats = tree.stats().unwrap();
        assert_eq!(stats.visits.iter().sum::<u32>(), 64);
        assert_eq!(stats.simulations, 64);
        assert!(stats.means.iter().all(|q| q.abs() <= 1.0));
    }
}
