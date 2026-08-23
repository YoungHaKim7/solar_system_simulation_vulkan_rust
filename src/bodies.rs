use crate::simulation::AU_DAY_GRAV_CONSTANT;

pub(crate) struct Spec {
    pub(crate) name: &'static str,
    pub(crate) mass: f64,
    pub(crate) semi_major_axis: f64,
    pub(crate) eccentricity: f64,
    pub(crate) display_radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) phase: f64,
}

pub(crate) const BODIES: [Spec; 9] = [
    Spec {
        name: "Sun",
        mass: 1.0,
        semi_major_axis: 0.0,
        eccentricity: 0.0,
        display_radius: 0.35,
        color: [1.00, 0.85, 0.40],
        phase: 0.0,
    },
    Spec {
        name: "Mercury",
        mass: 1.66e-7,
        semi_major_axis: 0.387,
        eccentricity: 0.206,
        display_radius: 0.055,
        color: [0.62, 0.58, 0.55],
        phase: 0.0,
    },
    Spec {
        name: "Venus",
        mass: 2.45e-6,
        semi_major_axis: 0.723,
        eccentricity: 0.007,
        display_radius: 0.085,
        color: [0.93, 0.82, 0.55],
        phase: 2.0,
    },
    Spec {
        name: "Earth",
        mass: 3.00e-6,
        semi_major_axis: 1.000,
        eccentricity: 0.017,
        display_radius: 0.09,
        color: [0.25, 0.45, 0.90],
        phase: 4.1,
    },
    Spec {
        name: "Mars",
        mass: 3.23e-7,
        semi_major_axis: 1.524,
        eccentricity: 0.093,
        display_radius: 0.07,
        color: [0.85, 0.40, 0.25],
        phase: 0.9,
    },
    Spec {
        name: "Jupiter",
        mass: 9.54e-4,
        semi_major_axis: 5.203,
        eccentricity: 0.048,
        display_radius: 0.22,
        color: [0.85, 0.72, 0.55],
        phase: 3.3,
    },
    Spec {
        name: "Saturn",
        mass: 2.86e-4,
        semi_major_axis: 9.537,
        eccentricity: 0.054,
        display_radius: 0.19,
        color: [0.90, 0.80, 0.55],
        phase: 5.4,
    },
    Spec {
        name: "Uranus",
        mass: 4.37e-5,
        semi_major_axis: 19.19,
        eccentricity: 0.047,
        display_radius: 0.13,
        color: [0.55, 0.80, 0.85],
        phase: 1.7,
    },
    Spec {
        name: "Neptune",
        mass: 5.15e-5,
        semi_major_axis: 30.07,
        eccentricity: 0.009,
        display_radius: 0.125,
        color: [0.35, 0.45, 0.90],
        phase: 2.8,
    },
];

pub(crate) fn orbital_periods() -> Vec<f64> {
    BODIES
        .iter()
        .skip(1)
        .map(|s| {
            2.0 * std::f64::consts::PI
                * (s.semi_major_axis.powi(3) / (AU_DAY_GRAV_CONSTANT * BODIES[0].mass)).sqrt()
        })
        .collect()
}
