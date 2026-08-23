use crate::bodies::BODIES;

pub(crate) const AU_DAY_GRAV_CONSTANT: f64 = 2.959122082855911e-4;
const MAX_SUBSTEP_DAYS: f64 = 0.2;

pub(crate) struct Body {
    pub(crate) mass: f64,
    pub(crate) pos: [f64; 2],
    pub(crate) vel: [f64; 2],
    pub(crate) acc: [f64; 2],
}

pub(crate) struct Simulation {
    pub(crate) bodies: Vec<Body>,
    pub(crate) t_days: f64,
    gravities: Vec<[f64; 2]>,
    previous_acc: Vec<[f64; 2]>,
}

impl Simulation {
    pub(crate) fn new() -> Self {
        let mut bodies = Vec::with_capacity(BODIES.len());
        for (i, spec) in BODIES.iter().enumerate() {
            if i == 0 {
                bodies.push(Body {
                    mass: spec.mass,
                    pos: [0.0, 0.0],
                    vel: [0.0, 0.0],
                    acc: [0.0, 0.0],
                });
            } else {
                let sun_mass = BODIES[0].mass;
                let r = spec.semi_major_axis * (1.0 + spec.eccentricity);
                let speed =
                    (AU_DAY_GRAV_CONSTANT * sun_mass * (2.0 / r - 1.0 / spec.semi_major_axis))
                        .sqrt();
                let (st, ct) = spec.phase.sin_cos();
                bodies.push(Body {
                    mass: spec.mass,
                    pos: [r * ct, r * st],
                    vel: [-speed * st, speed * ct],
                    acc: [0.0, 0.0],
                });
            }
        }
        let mut sim = Self {
            bodies,
            t_days: 0.0,
            gravities: vec![[0.0; 2]; BODIES.len()],
            previous_acc: vec![[0.0; 2]; BODIES.len()],
        };
        sim.compute_gravities();
        for (b, g) in sim.bodies.iter_mut().zip(&sim.gravities) {
            b.acc = *g;
        }
        sim
    }

    fn compute_gravities(&mut self) {
        let n = self.bodies.len();
        for g in self.gravities.iter_mut() {
            *g = [0.0; 2];
        }
        for i in 0..n {
            for j in i + 1..n {
                let pi = self.bodies[i].pos;
                let pj = self.bodies[j].pos;
                let dx = pj[0] - pi[0];
                let dy = pj[1] - pi[1];
                let r2 = dx * dx + dy * dy + 1e-9;
                let inv_r3 = 1.0 / (r2 * r2.sqrt());
                let gi = AU_DAY_GRAV_CONSTANT * self.bodies[j].mass * inv_r3;
                let gj = AU_DAY_GRAV_CONSTANT * self.bodies[i].mass * inv_r3;
                self.gravities[i][0] += dx * gi;
                self.gravities[i][1] += dy * gi;
                self.gravities[j][0] -= dx * gj;
                self.gravities[j][1] -= dy * gj;
            }
        }
    }

    pub(crate) fn advance(&mut self, days: f64) {
        if days <= 0.0 {
            return;
        }
        let steps = ((days / MAX_SUBSTEP_DAYS).ceil() as usize).clamp(1, 20000);
        let dt = days / steps as f64;
        for _ in 0..steps {
            for b in &mut self.bodies {
                b.pos[0] += b.vel[0] * dt + 0.5 * b.acc[0] * dt * dt;
                b.pos[1] += b.vel[1] * dt + 0.5 * b.acc[1] * dt * dt;
            }
            for (dst, b) in self.previous_acc.iter_mut().zip(&self.bodies) {
                *dst = b.acc;
            }
            self.compute_gravities();
            for (i, b) in self.bodies.iter_mut().enumerate() {
                b.vel[0] += 0.5 * (self.previous_acc[i][0] + self.gravities[i][0]) * dt;
                b.vel[1] += 0.5 * (self.previous_acc[i][1] + self.gravities[i][1]) * dt;
                b.acc = self.gravities[i];
            }
        }
        self.t_days += days;
    }
}
