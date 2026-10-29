//! Arena PUCT search used by the v2 self-play engine.
//!
//! Differences from `duo_search`: only the root position is stored (paths are
//! replayed with trusted moves), unexplored moves use first-play urgency (FPU)
//! reduction instead of a draw value, subtrees are reused between moves, and
//! terminal or solved positions back up exact values without the network.
use crate::duo::Game;

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    pub c_puct: f32,
    /// Unvisited children are valued at parent Q minus this × sqrt(visited prior mass).
    pub fpu_reduction: f32,
    pub fpu_root: f32,
    pub virtual_loss: f32,
    /// Utility = sign(margin) + score_weight × tanh(margin / score_scale).
    pub score_weight: f32,
    pub score_scale: f32,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            fpu_reduction: 0.25,
            fpu_root: 0.0,
            virtual_loss: 1.0,
            score_weight: 0.0,
            score_scale: 20.0,
        }
    }
}

impl SearchParams {
    /// Utility of a final score margin for the player it belongs to.
    pub fn utility(&self, margin: i32) -> f32 {
        margin.signum() as f32 + self.score_weight * (margin as f32 / self.score_scale).tanh()
    }
}

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Fresh,
    Pending,
    Expanded,
    Exact,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    visits: u32,
    virt: u32,
    /// Sum of backed-up values for the player who moved into this node.
    w: f32,
    /// For Exact nodes: the value for the player to move here.
    exact: f32,
    first_edge: u32,
    edge_count: u32,
    state: State,
}

impl Node {
    fn fresh() -> Self {
        Self {
            visits: 0,
            virt: 0,
            w: 0.0,
            exact: 0.0,
            first_edge: 0,
            edge_count: 0,
            state: State::Fresh,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    action: u16,
    child: u32,
    prior: f32,
}

/// A position awaiting a network evaluation. `actions` is its legal list.
#[derive(Clone, Debug)]
pub struct Leaf {
    pub path: Vec<u32>,
    pub game: Game,
    pub actions: Vec<usize>,
}

#[allow(clippy::large_enum_variant)] // Leaves are moved straight into the pending list.
pub enum Selection {
    Leaf(Leaf),
    /// A terminal or solved position was reached and backed up immediately.
    Exact,
    /// The path ran into a leaf already waiting for the network.
    Collision,
}

#[derive(Clone, Debug)]
pub struct RootChild {
    pub action: usize,
    pub visits: u32,
    pub prior: f32,
    /// Mean value for the player to move at the root (0 when unvisited).
    pub q: f32,
}

#[derive(Clone, Debug)]
pub struct Tree {
    root_game: Game,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Tree {
    pub fn new(game: Game) -> Self {
        Self {
            root_game: game,
            nodes: vec![Node::fresh()],
            edges: Vec::new(),
        }
    }

    pub fn root_game(&self) -> &Game {
        &self.root_game
    }
    pub fn root_expanded(&self) -> bool {
        self.nodes[0].state == State::Expanded
    }
    pub fn root_visits(&self) -> u32 {
        self.nodes[0].visits
    }
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    /// Root value for the player to move, averaged over the whole search.
    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[0];
        if root.state == State::Exact {
            root.exact
        } else if root.visits == 0 {
            0.0
        } else {
            -root.w / root.visits as f32
        }
    }

    /// Descend from the root once. `solver` may prove a fresh position's exact
    /// margin for its player to move; such nodes are never sent to the network.
    pub fn select(
        &mut self,
        params: &SearchParams,
        solver: &mut dyn FnMut(&Game) -> Option<i32>,
    ) -> Selection {
        let mut game = self.root_game.clone();
        let mut path = vec![0_u32];
        let mut current = 0_usize;
        loop {
            match self.nodes[current].state {
                State::Exact => {
                    let value = self.nodes[current].exact;
                    self.backup(&path, value, false);
                    return Selection::Exact;
                }
                State::Pending => return Selection::Collision,
                State::Fresh => {
                    let exact = if game.is_terminal() {
                        let scores = game.scores();
                        let me = game.turn();
                        Some(i32::from(scores[me]) - i32::from(scores[1 - me]))
                    } else if current != 0 {
                        // The root is solved (with its best move) before search starts.
                        solver(&game)
                    } else {
                        None
                    };
                    if let Some(margin) = exact {
                        let node = &mut self.nodes[current];
                        node.state = State::Exact;
                        node.exact = params.utility(margin);
                        let value = node.exact;
                        self.backup(&path, value, false);
                        return Selection::Exact;
                    }
                    self.nodes[current].state = State::Pending;
                    for &node in &path {
                        self.nodes[node as usize].virt += 1;
                    }
                    let actions = game.legal_actions();
                    return Selection::Leaf(Leaf {
                        path,
                        game,
                        actions,
                    });
                }
                State::Expanded => {
                    let edge = self.pick(current, params, current == 0);
                    let index = (self.nodes[current].first_edge + edge) as usize;
                    let mut child = self.edges[index].child;
                    if child == NONE {
                        child = self.nodes.len() as u32;
                        self.nodes.push(Node::fresh());
                        self.edges[index].child = child;
                    }
                    game.apply_trusted(usize::from(self.edges[index].action));
                    path.push(child);
                    current = child as usize;
                }
            }
        }
    }

    fn pick(&self, node: usize, params: &SearchParams, root: bool) -> u32 {
        let x = &self.nodes[node];
        let edges = &self.edges[x.first_edge as usize..(x.first_edge + x.edge_count) as usize];
        let parent_q = if x.visits > 0 {
            -x.w / x.visits as f32
        } else {
            0.0
        };
        let mut visited_mass = 0.0;
        for e in edges {
            if e.child != NONE {
                let c = &self.nodes[e.child as usize];
                if c.visits + c.virt > 0 {
                    visited_mass += e.prior;
                }
            }
        }
        let reduction = if root {
            params.fpu_root
        } else {
            params.fpu_reduction
        };
        let fpu = parent_q - reduction * visited_mass.sqrt();
        let total = (x.visits + x.virt).saturating_sub(1);
        let scale = params.c_puct * ((total + 1) as f32).sqrt();
        let mut best = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, e) in edges.iter().enumerate() {
            let (n, q) = if e.child == NONE {
                (0, fpu)
            } else {
                let c = &self.nodes[e.child as usize];
                let n = c.visits + c.virt;
                if n == 0 {
                    (0, fpu)
                } else {
                    (n, (c.w - params.virtual_loss * c.virt as f32) / n as f32)
                }
            };
            let score = q + scale * e.prior / (1 + n) as f32;
            if score > best_score {
                best_score = score;
                best = i as u32;
            }
        }
        best
    }

    /// `value` is for the player to move at the last node of `path`.
    fn backup(&mut self, path: &[u32], value: f32, virtual_path: bool) {
        let mut v = -value;
        for &node in path.iter().rev() {
            let n = &mut self.nodes[node as usize];
            n.visits += 1;
            n.w += v;
            if virtual_path {
                n.virt -= 1;
            }
            v = -v;
        }
    }

    /// Store the network's priors (aligned with `leaf.actions`) and back up its value.
    pub fn expand(&mut self, leaf: &Leaf, priors: &[f32], value: f32) {
        let node = *leaf.path.last().unwrap() as usize;
        assert_eq!(self.nodes[node].state, State::Pending);
        assert_eq!(priors.len(), leaf.actions.len());
        let sum: f32 = priors.iter().sum();
        let uniform = 1.0 / priors.len() as f32;
        let first = self.edges.len() as u32;
        self.edges
            .extend(leaf.actions.iter().zip(priors).map(|(&action, &p)| Edge {
                action: action as u16,
                child: NONE,
                prior: if sum > 0.0 && sum.is_finite() {
                    p / sum
                } else {
                    uniform
                },
            }));
        let n = &mut self.nodes[node];
        n.first_edge = first;
        n.edge_count = leaf.actions.len() as u32;
        n.state = State::Expanded;
        self.backup(&leaf.path, value, true);
    }

    /// Mix exploration noise into the expanded root's priors.
    pub fn mix_root_noise(&mut self, noise: &[f32], weight: f32) {
        let root = self.nodes[0];
        assert_eq!(root.state, State::Expanded);
        let edges =
            &mut self.edges[root.first_edge as usize..(root.first_edge + root.edge_count) as usize];
        for (edge, &n) in edges.iter_mut().zip(noise) {
            edge.prior = (1.0 - weight) * edge.prior + weight * n;
        }
    }

    pub fn root_edge_count(&self) -> usize {
        self.nodes[0].edge_count as usize
    }

    pub fn root_children(&self) -> Vec<RootChild> {
        let root = &self.nodes[0];
        if root.state != State::Expanded {
            return vec![];
        }
        self.edges[root.first_edge as usize..(root.first_edge + root.edge_count) as usize]
            .iter()
            .map(|e| {
                let (visits, q) = if e.child == NONE {
                    (0, 0.0)
                } else {
                    let c = &self.nodes[e.child as usize];
                    (
                        c.visits,
                        if c.visits > 0 {
                            c.w / c.visits as f32
                        } else {
                            0.0
                        },
                    )
                };
                RootChild {
                    action: usize::from(e.action),
                    visits,
                    prior: e.prior,
                    q,
                }
            })
            .collect()
    }

    /// Move the root along `action`, keeping that subtree when it exists.
    pub fn advance(&mut self, action: usize) {
        let root = self.nodes[0];
        let mut keep = NONE;
        if root.state == State::Expanded {
            for e in
                &self.edges[root.first_edge as usize..(root.first_edge + root.edge_count) as usize]
            {
                if usize::from(e.action) == action {
                    keep = e.child;
                }
            }
        }
        self.root_game.apply_trusted(action);
        if keep == NONE
            || matches!(
                self.nodes[keep as usize].state,
                State::Fresh | State::Pending
            )
        {
            self.nodes = vec![Node::fresh()];
            self.edges.clear();
            return;
        }
        let mut nodes = vec![self.nodes[keep as usize]];
        let mut edges = Vec::new();
        let mut order = vec![keep];
        let mut i = 0;
        while i < order.len() {
            let old = self.nodes[order[i] as usize];
            if old.state == State::Expanded {
                let first = edges.len() as u32;
                for e in
                    &self.edges[old.first_edge as usize..(old.first_edge + old.edge_count) as usize]
                {
                    let mut copy = *e;
                    if e.child != NONE {
                        copy.child = nodes.len() as u32;
                        nodes.push(self.nodes[e.child as usize]);
                        order.push(e.child);
                    }
                    edges.push(copy);
                }
                nodes[i].first_edge = first;
            }
            i += 1;
        }
        self.nodes = nodes;
        self.edges = edges;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::duo::PASS;

    fn no_solver(_: &Game) -> Option<i32> {
        None
    }

    /// Uniform priors and zero values: a pure visit-accounting check.
    fn run(tree: &mut Tree, sims: usize, batch: usize) -> usize {
        let params = SearchParams::default();
        let mut done = 0;
        while done < sims {
            let mut leaves = Vec::new();
            while leaves.len() < batch && done < sims {
                match tree.select(&params, &mut no_solver) {
                    Selection::Leaf(l) => {
                        leaves.push(l);
                        done += 1;
                    }
                    Selection::Exact => done += 1,
                    Selection::Collision => break,
                }
            }
            for leaf in &leaves {
                let priors = vec![1.0; leaf.actions.len()];
                tree.expand(leaf, &priors, 0.0);
            }
        }
        done
    }

    #[test]
    fn visits_add_up_and_virtual_loss_clears() {
        let mut tree = Tree::new(Game::new());
        run(&mut tree, 300, 8);
        let children = tree.root_children();
        let sum: u32 = children.iter().map(|c| c.visits).sum();
        assert_eq!(sum + 1, tree.root_visits());
        assert!(tree.nodes.iter().all(|n| n.virt == 0));
    }

    #[test]
    fn reuse_keeps_the_chosen_subtree() {
        let mut tree = Tree::new(Game::new());
        run(&mut tree, 400, 4);
        let best = tree
            .root_children()
            .into_iter()
            .max_by_key(|c| c.visits)
            .unwrap();
        tree.advance(best.action);
        assert_eq!(tree.root_visits(), best.visits);
        let sum: u32 = tree.root_children().iter().map(|c| c.visits).sum();
        assert_eq!(sum + 1, tree.root_visits());
        run(&mut tree, 100, 4);
        assert!(tree.nodes.iter().all(|n| n.virt == 0));
    }

    #[test]
    fn exact_endings_need_no_network() {
        // Play random moves until near the end, then search to the terminal.
        let mut game = Game::new();
        let mut state = 3_u64;
        while game.legal_actions_for(0) != vec![PASS] || game.legal_actions_for(1).len() > 3 {
            if game.is_terminal() {
                break;
            }
            let actions = game.legal_actions();
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            game.step(actions[(state >> 33) as usize % actions.len()])
                .unwrap();
        }
        if game.is_terminal() {
            return;
        }
        let mut tree = Tree::new(game);
        run(&mut tree, 200, 2);
        assert!(tree.root_visits() > 1);
    }
}
