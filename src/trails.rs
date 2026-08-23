use std::collections::VecDeque;

use crate::{bodies::BODIES, simulation::Body, vertices::TrailVertex};

pub(crate) const TRAIL_LEN: usize = 700;

pub(crate) struct Trails {
    queues: Vec<VecDeque<TrailVertex>>,
    next_sample: Vec<f64>,
    periods: Vec<f64>,
    scratch: Vec<TrailVertex>,
}

fn trail_vertex(i: usize, b: &Body) -> TrailVertex {
    TrailVertex {
        pos: [b.pos[0] as f32, b.pos[1] as f32],
        color: [
            BODIES[i].color[0] * 0.85,
            BODIES[i].color[1] * 0.85,
            BODIES[i].color[2] * 0.85,
        ],
        alpha: 0.0,
    }
}

impl Trails {
    pub(crate) fn new(periods: Vec<f64>) -> Self {
        Self {
            queues: Vec::new(),
            next_sample: Vec::new(),
            periods,
            scratch: Vec::new(),
        }
    }

    pub(crate) fn periods(&self) -> &[f64] {
        &self.periods
    }

    pub(crate) fn len(&self) -> usize {
        self.queues.len()
    }

    pub(crate) fn reset(&mut self, bodies: &[Body]) {
        let n = bodies.len();
        self.next_sample = vec![f64::INFINITY; n];
        self.next_sample[0] = 0.0;

        self.queues = bodies
            .iter()
            .enumerate()
            .map(|(i, b)| VecDeque::from(vec![trail_vertex(i, b); TRAIL_LEN]))
            .collect();
    }

    pub(crate) fn sample(&mut self, t_days: f64, bodies: &[Body]) {
        for (i, body) in bodies.iter().enumerate().take(self.queues.len()).skip(1) {
            if t_days < self.next_sample[i] {
                continue;
            }
            let v = trail_vertex(i, body);
            let dq = &mut self.queues[i];
            dq.push_back(v);
            if dq.len() > TRAIL_LEN {
                dq.pop_front();
            }
            self.next_sample[i] = t_days + self.periods[i - 1] / 260.0;
        }
    }

    pub(crate) fn flatten(&mut self) -> &[TrailVertex] {
        self.scratch.clear();
        for dq in &self.queues {
            let n = dq.len().max(1) as f32;
            for (k, v) in dq.iter().rev().enumerate() {
                let mut v = *v;
                v.alpha = 0.65 * (k + 1) as f32 / n;
                self.scratch.push(v);
            }
        }
        &self.scratch
    }
}
