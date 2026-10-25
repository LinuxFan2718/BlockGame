use crate::Observation;

/// A policy maps an observation to an action. Learning will live outside the rules.
pub trait Agent {
    /// Called on the agent's turn (`to_move == 1`). Return None if unable to act.
    /// The engine still validates the chosen action; the mask is advisory input.
    fn select_action(&mut self, observation: &Observation) -> Option<usize>;
}

/// Uniformly samples legal actions. Same seed + same inputs = same actions.
/// This is a baseline opponent, not a learning agent.
#[derive(Debug, Clone)]
pub struct RandomAgent {
    state: u64,
}

impl RandomAgent {
    /// Every seed, including zero, is valid.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    // SplitMix64, adapted from Sebastiano Vigna's public-domain reference:
    // https://prng.di.unimi.it/splitmix64.c
    // Explicit wrapping keeps debug and release builds identical.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn sample_below(&mut self, bound: u64) -> usize {
        // Reject the short remainder so modulo does not bias the distribution.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

impl Agent for RandomAgent {
    fn select_action(&mut self, observation: &Observation) -> Option<usize> {
        if observation.to_move != 1 {
            return None;
        }
        let count = observation
            .legal_mask
            .iter()
            .filter(|&&legal| legal)
            .count();
        if count == 0 {
            return None;
        }
        let choice = self.sample_below(count as u64);
        observation
            .legal_mask
            .iter()
            .enumerate()
            .filter_map(|(action, &legal)| legal.then_some(action))
            .nth(choice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_matches_reference_sequence() {
        let mut bot = RandomAgent::new(0);
        assert_eq!(bot.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(bot.next_u64(), 0x6e78_9e6a_a1b9_65f4);
    }
}
