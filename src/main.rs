mod app;
mod bodies;
mod camera;
mod debug;
mod gpu;
mod renderer;
mod shaders;
mod simulation;
mod trails;
mod vertices;

use std::error::Error;
use winit::event_loop::EventLoop;

use crate::app::App;

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
