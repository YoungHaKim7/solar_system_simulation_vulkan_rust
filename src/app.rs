use std::time::Instant;

use vulkano::{
    Validated, VulkanError,
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    swapchain::{SwapchainPresentInfo, acquire_next_image},
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::WindowId,
};

use crate::{
    bodies::{BODIES, orbital_periods},
    camera::{Camera, START_VIEW_HEIGHT},
    gpu::Gpu,
    renderer::RenderContext,
    simulation::Simulation,
    trails::Trails,
    vertices::{BodyVertex, PushConstants, fill_body_vertices},
};

const MIN_SPEED: f64 = 0.01;
const MAX_SPEED: f64 = 20000.0;
const DEFAULT_SPEED: f64 = 20.0;

pub(crate) struct App {
    pub(crate) gpu: Gpu,
    pub(crate) sim: Simulation,
    pub(crate) camera: Camera,
    paused: bool,
    sim_speed: f64,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    last_frame: Instant,
    pub(crate) trails: Trails,
    pub(crate) body_scratch: Vec<BodyVertex>,
    debug_done: bool,
    pub(crate) rcx: Option<RenderContext>,
}

impl App {
    pub(crate) fn new(event_loop: &EventLoop<()>) -> Self {
        println!("Solar system simulation");
        println!(
            "Controls: drag = pan | scroll = zoom | space = pause | up/down = time speed | R = reset | Esc = quit"
        );

        let gpu = Gpu::new(event_loop);

        let mut app = App {
            gpu,
            sim: Simulation::new(),
            camera: Camera {
                center: [0.0, 0.0],
                height: START_VIEW_HEIGHT,
            },
            paused: false,
            sim_speed: DEFAULT_SPEED,
            dragging: false,
            last_cursor: None,
            last_frame: Instant::now(),
            trails: Trails::new(orbital_periods()),
            body_scratch: Vec::new(),
            debug_done: false,
            rcx: None,
        };

        app.reset_universe();

        for (i, spec) in BODIES.iter().enumerate() {
            if i == 0 {
                println!("{:>8}: central star", spec.name);
            } else {
                println!(
                    "{:>8}: period {:.1} days",
                    spec.name,
                    app.trails.periods()[i - 1]
                );
            }
        }

        app
    }

    fn reset_universe(&mut self) {
        self.sim = Simulation::new();
        self.camera.center = [0.0; 2];
        self.camera.height = START_VIEW_HEIGHT;

        self.trails.reset(&self.sim.bodies);
        self.trails.sample(self.sim.t_days, &self.sim.bodies);
        fill_body_vertices(&self.sim.bodies, &mut self.body_scratch);
        self.trails.flatten();
    }

    fn set_title(&self) {
        if let Some(rcx) = &self.rcx {
            let speed_label = if self.paused {
                "paused".to_string()
            } else {
                format!("{:.1} days/s", self.sim_speed)
            };
            let title = format!(
                "Solar System [{speed_label}] — drag: pan · scroll: zoom · space: pause · up/down: speed · R: reset · Esc: quit"
            );
            rcx.window.set_title(&title);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.rcx = Some(RenderContext::new(
            event_loop,
            &self.gpu.instance,
            &self.gpu.device,
            &self.gpu.memory_allocator,
            self.body_scratch.clone(),
            self.trails.flatten().to_vec(),
        ));

        self.set_title();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain = true;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard(event, event_loop);
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = (position.x, position.y);
                if self.dragging
                    && let Some(prev) = self.last_cursor
                {
                    let height_px = self
                        .rcx
                        .as_ref()
                        .map(|r| r.window.inner_size().height as f64)
                        .unwrap_or(800.0);
                    self.camera
                        .pan(new_pos.0 - prev.0, new_pos.1 - prev.1, height_px);
                }
                self.last_cursor = Some(new_pos);
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    self.dragging = state == ElementState::Pressed;
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let factor = match delta {
                    MouseScrollDelta::LineDelta(_, y) => 1.15f64.powf(y as f64),
                    MouseScrollDelta::PixelDelta(p) => 1.01f64.powf(p.y),
                };
                if factor.is_finite()
                    && factor != 1.0
                    && let Some(cursor) = self.last_cursor
                    && let Some(rcx) = self.rcx.as_ref()
                {
                    let size = rcx.window.inner_size();
                    self.camera.zoom(
                        factor,
                        cursor.0,
                        cursor.1,
                        size.width as f64,
                        size.height as f64,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if !self.debug_done && std::env::var_os("SOLAR_DUMP_FRAME").is_some() {
            self.debug_done = true;
            self.debug_dump_frame("/tmp/opencode/solar_frame.ppm");
            event_loop.exit();
            return;
        }
        if let Some(rcx) = self.rcx.as_ref() {
            rcx.window.request_redraw();
        }
    }
}

impl App {
    fn handle_keyboard(&mut self, event: KeyEvent, event_loop: &ActiveEventLoop) {
        if event.state != ElementState::Pressed {
            return;
        }
        match event.logical_key {
            Key::Named(NamedKey::Space) => {
                self.paused = !self.paused;
                self.last_frame = Instant::now();
                self.set_title();
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.sim_speed = (self.sim_speed * 1.5).min(MAX_SPEED);
                self.set_title();
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.sim_speed = (self.sim_speed / 1.5).max(MIN_SPEED);
                self.set_title();
            }
            Key::Character(c) if c.eq_ignore_ascii_case("r") => {
                self.reset_universe();
                self.last_frame = Instant::now();
                self.set_title();
            }
            Key::Named(NamedKey::Escape) => {
                event_loop.exit();
            }
            _ => {}
        }
    }

    fn redraw(&mut self) {
        let window_size = match self.rcx.as_ref() {
            Some(rcx) => rcx.window.inner_size(),
            None => return,
        };

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        let now = Instant::now();
        let dt_real = now.duration_since(self.last_frame).as_secs_f64().min(0.05);
        self.last_frame = now;

        if !self.paused {
            self.sim.advance(self.sim_speed * dt_real);
            self.trails.sample(self.sim.t_days, &self.sim.bodies);
        }

        let (w, h) = (window_size.width as f64, window_size.height as f64);
        let scale = self.camera.scales(w, h);
        let offset = self.camera.offsets(scale);
        let sun_pos = self.sim.bodies[0].pos;
        let push = PushConstants {
            transform: [
                scale[0] as f32,
                scale[1] as f32,
                offset[0] as f32,
                offset[1] as f32,
            ],
            params: [sun_pos[0] as f32, sun_pos[1] as f32, 0.0, 0.0],
        };

        fill_body_vertices(&self.sim.bodies, &mut self.body_scratch);
        let body_vertex_count = self.body_scratch.len() as u32;
        let trail_strips = self.trails.len() as u64;
        let trail_vertices = self.trails.flatten();

        let rcx = self.rcx.as_mut().unwrap();

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if rcx.recreate_swapchain {
            rcx.refresh_swapchain(window_size.into());
        }

        if let Some(previous) = rcx.previous_frame_end.take() {
            drop(previous);
        }

        {
            let mut guard = rcx.body_vertex_buffer.write().unwrap();
            guard.copy_from_slice(&self.body_scratch);
        }
        {
            let mut guard = rcx.trail_vertex_buffer.write().unwrap();
            guard.copy_from_slice(trail_vertices);
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    rcx.previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        rcx.record_scene(
            &mut builder,
            rcx.attachment_image_views[image_index as usize].clone(),
            rcx.viewport.clone(),
            push,
            trail_strips,
            body_vertex_count,
        );

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.gpu.device.clone())
            .join(acquire_future)
            .then_execute(self.gpu.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.gpu.queue.clone(),
                SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());
            }
        }
    }
}
