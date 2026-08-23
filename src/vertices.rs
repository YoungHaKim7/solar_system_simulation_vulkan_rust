use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::{bodies::BODIES, simulation::Body};

pub(crate) const QUAD_CORNERS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
pub(crate) struct BodyVertex {
    #[format(R32G32_SFLOAT)]
    pub(crate) center: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub(crate) corner: [f32; 2],
    #[format(R32_SFLOAT)]
    pub(crate) radius: f32,
    #[format(R32G32B32_SFLOAT)]
    pub(crate) color: [f32; 3],
    #[format(R32_SFLOAT)]
    pub(crate) glow: f32,
}

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
pub(crate) struct TrailVertex {
    #[format(R32G32_SFLOAT)]
    pub(crate) pos: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    pub(crate) color: [f32; 3],
    #[format(R32_SFLOAT)]
    pub(crate) alpha: f32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
pub(crate) struct PushConstants {
    pub(crate) transform: [f32; 4],
    pub(crate) params: [f32; 4],
}

pub(crate) fn fill_body_vertices(bodies: &[Body], out: &mut Vec<BodyVertex>) {
    out.clear();
    for (i, b) in bodies.iter().enumerate() {
        for corner in QUAD_CORNERS {
            out.push(BodyVertex {
                center: [b.pos[0] as f32, b.pos[1] as f32],
                corner,
                radius: BODIES[i].display_radius,
                color: BODIES[i].color,
                glow: if i == 0 { 1.0 } else { 0.0 },
            });
        }
    }
}
