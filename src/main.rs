use std::{collections::VecDeque, error::Error, sync::Arc, time::Instant};
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo,
        RenderingAttachmentInfo, RenderingInfo, allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo,
        QueueFlags, physical::PhysicalDeviceType,
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
            },
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowId},
};

const AU_DAY_GRAV_CONSTANT: f64 = 2.959122082855911e-4;
const TRAIL_LEN: usize = 700;
const MAX_SUBSTEP_DAYS: f64 = 0.2;
const START_VIEW_HEIGHT: f64 = 70.0;
const MIN_SPEED: f64 = 0.01;
const MAX_SPEED: f64 = 20000.0;
const DEFAULT_SPEED: f64 = 20.0;

const QUAD_CORNERS: [[f32; 2]; 6] = [
    [-1.0, -1.0],
    [1.0, -1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, 1.0],
    [-1.0, 1.0],
];

struct Spec {
    name: &'static str,
    mass: f64,
    semi_major_axis: f64,
    eccentricity: f64,
    display_radius: f32,
    color: [f32; 3],
    phase: f64,
}

const BODIES: [Spec; 9] = [
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

mod body_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "../assets/body.vert",
    }
}

mod body_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "../assets/body.frag",
    }
}

mod line_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "../assets/line.vert",
    }
}

mod line_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "../assets/line.frag"
    }
}

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
struct BodyVertex {
    #[format(R32G32_SFLOAT)]
    center: [f32; 2],
    #[format(R32G32_SFLOAT)]
    corner: [f32; 2],
    #[format(R32_SFLOAT)]
    radius: f32,
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
    #[format(R32_SFLOAT)]
    glow: f32,
}

#[derive(BufferContents, Clone, Copy, Vertex)]
#[repr(C)]
struct TrailVertex {
    #[format(R32G32_SFLOAT)]
    pos: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
    #[format(R32_SFLOAT)]
    alpha: f32,
}

#[derive(BufferContents, Clone, Copy)]
#[repr(C)]
struct PushConstants {
    transform: [f32; 4],
    params: [f32; 4],
}

struct Camera {
    center: [f64; 2],
    height: f64,
}

impl Camera {
    fn scales(&self, width: f64, height_px: f64) -> [f64; 2] {
        let aspect = width / height_px;
        [2.0 / (self.height * aspect), 2.0 / self.height]
    }

    fn offsets(&self, s: [f64; 2]) -> [f64; 2] {
        [-self.center[0] * s[0], -self.center[1] * s[1]]
    }

    fn screen_to_world(&self, px: f64, py: f64, width: f64, height_px: f64) -> [f64; 2] {
        let s = self.scales(width, height_px);
        let o = self.offsets(s);
        let nx = px / width * 2.0 - 1.0;
        let ny = py / height_px * 2.0 - 1.0;
        [(nx - o[0]) / s[0], (-ny - o[1]) / s[1]]
    }

    fn zoom(&mut self, factor: f64, px: f64, py: f64, width: f64, height_px: f64) {
        let before = self.screen_to_world(px, py, width, height_px);
        self.height = (self.height / factor).clamp(0.15, 600.0);
        let after = self.screen_to_world(px, py, width, height_px);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    fn pan(&mut self, dx_px: f64, dy_px: f64, height_px: f64) {
        let world_per_px = self.height / height_px;
        self.center[0] -= dx_px * world_per_px;
        self.center[1] += dy_px * world_per_px;
    }
}

struct Body {
    mass: f64,
    pos: [f64; 2],
    vel: [f64; 2],
    acc: [f64; 2],
}

struct Simulation {
    bodies: Vec<Body>,
    t_days: f64,
    gravities: Vec<[f64; 2]>,
    previous_acc: Vec<[f64; 2]>,
}

impl Simulation {
    fn new() -> Self {
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

    fn advance(&mut self, days: f64) {
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

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    sim: Simulation,
    camera: Camera,
    paused: bool,
    sim_speed: f64,
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    last_frame: Instant,
    periods: Vec<f64>,
    trails: Vec<VecDeque<TrailVertex>>,
    next_sample: Vec<f64>,
    trail_scratch: Vec<TrailVertex>,
    body_scratch: Vec<BodyVertex>,
    debug_done: bool,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    attachment_image_views: Vec<Arc<ImageView>>,
    body_pipeline: Arc<GraphicsPipeline>,
    line_pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    body_vertex_buffer: Subbuffer<[BodyVertex]>,
    trail_vertex_buffer: Subbuffer<[TrailVertex]>,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        println!("Solar system simulation");
        println!(
            "Controls: drag = pan | scroll = zoom | space = pause | up/down = time speed | R = reset | Esc = quit"
        );

        let library = unsafe { VulkanLibrary::new() }.unwrap();

        let required_extensions = Surface::required_extensions(event_loop);

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: &device_extensions,
                enabled_features: &DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();

        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        let mut app = App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            memory_allocator,
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
            periods: Vec::new(),
            trails: Vec::new(),
            next_sample: Vec::new(),
            trail_scratch: Vec::new(),
            body_scratch: Vec::new(),
            debug_done: false,
            rcx: None,
        };

        app.periods = BODIES
            .iter()
            .skip(1)
            .map(|s| {
                2.0 * std::f64::consts::PI
                    * (s.semi_major_axis.powi(3) / (AU_DAY_GRAV_CONSTANT * BODIES[0].mass)).sqrt()
            })
            .collect();

        app.reset_universe();

        for (i, spec) in BODIES.iter().enumerate() {
            if i == 0 {
                println!("{:>8}: central star", spec.name);
            } else {
                println!("{:>8}: period {:.1} days", spec.name, app.periods[i - 1]);
            }
        }

        app
    }

    fn reset_universe(&mut self) {
        self.sim = Simulation::new();
        self.camera.center = [0.0; 2];
        self.camera.height = START_VIEW_HEIGHT;

        let n = self.sim.bodies.len();
        self.next_sample = vec![f64::INFINITY; n];
        self.next_sample[0] = 0.0;

        self.trails = self
            .sim
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let v = TrailVertex {
                    pos: [b.pos[0] as f32, b.pos[1] as f32],
                    color: [
                        BODIES[i].color[0] * 0.85,
                        BODIES[i].color[1] * 0.85,
                        BODIES[i].color[2] * 0.85,
                    ],
                    alpha: 0.0,
                };
                VecDeque::from(vec![v; TRAIL_LEN])
            })
            .collect();

        self.sample_trails();
        self.build_body_vertices();
        self.flatten_trails();
    }

    fn sample_trails(&mut self) {
        let t = self.sim.t_days;
        for i in 1..self.trails.len() {
            if t < self.next_sample[i] {
                continue;
            }
            let b = &self.sim.bodies[i];
            let v = TrailVertex {
                pos: [b.pos[0] as f32, b.pos[1] as f32],
                color: [
                    BODIES[i].color[0] * 0.85,
                    BODIES[i].color[1] * 0.85,
                    BODIES[i].color[2] * 0.85,
                ],
                alpha: 0.0,
            };
            let dq = &mut self.trails[i];
            dq.push_back(v);
            if dq.len() > TRAIL_LEN {
                dq.pop_front();
            }
            self.next_sample[i] = t + self.periods[i - 1] / 260.0;
        }
    }

    fn flatten_trails(&mut self) {
        self.trail_scratch.clear();
        for dq in &self.trails {
            let n = dq.len().max(1) as f32;
            for (k, v) in dq.iter().rev().enumerate() {
                let mut v = *v;
                v.alpha = 0.65 * (k + 1) as f32 / n;
                self.trail_scratch.push(v);
            }
        }
    }

    fn build_body_vertices(&mut self) {
        self.body_scratch.clear();
        for (i, b) in self.sim.bodies.iter().enumerate() {
            for corner in QUAD_CORNERS {
                self.body_scratch.push(BodyVertex {
                    center: [b.pos[0] as f32, b.pos[1] as f32],
                    corner,
                    radius: BODIES[i].display_radius,
                    color: BODIES[i].color,
                    glow: if i == 0 { 1.0 } else { 0.0 },
                });
            }
        }
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
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_inner_size(LogicalSize::new(1280.0, 800.0)),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            Swapchain::new(
                &self.device,
                &surface,
                &SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let attachment_image_views = images
            .iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();

        let body_pipeline = {
            let vs = unsafe { body_vs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { body_fs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = BodyVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            let layout = PipelineLayout::from_stages(&self.device, &stages).unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: &[Some(swapchain.image_format())],
                ..Default::default()
            };

            GraphicsPipeline::new(
                &self.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState::default()),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::SrcAlpha,
                                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                                alpha_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            }),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some((&subpass).into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        let line_pipeline = {
            let vs = unsafe { line_vs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { line_fs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = TrailVertex::per_vertex().definition(&vs).unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];

            let layout = PipelineLayout::from_stages(&self.device, &stages).unwrap();

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: &[Some(swapchain.image_format())],
                ..Default::default()
            };

            GraphicsPipeline::new(
                &self.device,
                None,
                &GraphicsPipelineCreateInfo {
                    stages: &stages,
                    vertex_input_state: Some(&vertex_input_state),
                    input_assembly_state: Some(&InputAssemblyState {
                        topology: PrimitiveTopology::LineStrip,
                        ..Default::default()
                    }),
                    viewport_state: Some(&ViewportState::default()),
                    rasterization_state: Some(&RasterizationState::default()),
                    multisample_state: Some(&MultisampleState::default()),
                    color_blend_state: Some(&ColorBlendState {
                        attachments: &[ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend {
                                color_blend_op: BlendOp::Add,
                                src_color_blend_factor: BlendFactor::SrcAlpha,
                                dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                                alpha_blend_op: BlendOp::Add,
                                src_alpha_blend_factor: BlendFactor::One,
                                dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
                            }),
                            ..Default::default()
                        }],
                        ..Default::default()
                    }),
                    dynamic_state: &[DynamicState::Viewport],
                    subpass: Some((&subpass).into()),
                    ..GraphicsPipelineCreateInfo::new(&layout)
                },
            )
            .unwrap()
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let body_vertex_buffer = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.body_scratch.clone(),
        )
        .unwrap();

        let trail_vertex_buffer = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.trail_scratch.clone(),
        )
        .unwrap();

        let recreate_swapchain = false;
        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            attachment_image_views,
            body_pipeline,
            line_pipeline,
            viewport,
            body_vertex_buffer,
            trail_vertex_buffer,
            recreate_swapchain,
            previous_frame_end,
        });

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
                if self.dragging {
                    if let Some(prev) = self.last_cursor {
                        let height_px = self
                            .rcx
                            .as_ref()
                            .map(|r| r.window.inner_size().height as f64)
                            .unwrap_or(800.0);
                        self.camera
                            .pan(new_pos.0 - prev.0, new_pos.1 - prev.1, height_px);
                    }
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
                    MouseScrollDelta::PixelDelta(p) => 1.01f64.powf(p.y as f64),
                };
                if factor.is_finite() && factor != 1.0 {
                    if let Some(cursor) = self.last_cursor {
                        if let Some(rcx) = self.rcx.as_ref() {
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
            self.sample_trails();
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

        self.build_body_vertices();
        self.flatten_trails();
        let body_vertex_count = self.body_scratch.len() as u32;
        let trail_len = TRAIL_LEN as u32;
        let trail_bodies = self.trails.len() as u64;

        let rcx = self.rcx.as_mut().unwrap();

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if rcx.recreate_swapchain {
            let (new_swapchain, new_images) = rcx
                .swapchain
                .recreate(&SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            rcx.swapchain = new_swapchain;
            rcx.attachment_image_views = new_images
                .iter()
                .map(|image| ImageView::new_default(image).unwrap())
                .collect::<Vec<_>>();
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
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
            guard.copy_from_slice(&self.trail_scratch);
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.012, 0.012, 0.03, 1.0].into()),
                    ..RenderingAttachmentInfo::new(
                        rcx.attachment_image_views[image_index as usize].clone(),
                    )
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(rcx.line_pipeline.clone())
            .unwrap()
            .push_constants(rcx.line_pipeline.layout().clone(), 0, push)
            .unwrap();

        for i in 0..trail_bodies {
            let start = i * TRAIL_LEN as u64;
            let slice = rcx
                .trail_vertex_buffer
                .clone()
                .slice(start..start + TRAIL_LEN as u64);
            builder.bind_vertex_buffers(0, slice).unwrap();
            unsafe { builder.draw(trail_len, 1, 0, 0) }.unwrap();
        }

        builder
            .bind_pipeline_graphics(rcx.body_pipeline.clone())
            .unwrap()
            .push_constants(rcx.body_pipeline.layout().clone(), 0, push)
            .unwrap()
            .bind_vertex_buffers(0, rcx.body_vertex_buffer.clone())
            .unwrap();

        unsafe { builder.draw(body_vertex_count, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();

        let command_buffer = builder.build().unwrap();

        let future = sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
    fn debug_dump_frame(&mut self, path: &str) {
        self.sim.advance(365.25 * 3.0);
        self.sample_trails();
        self.build_body_vertices();
        self.flatten_trails();

        let width = 1280u32;
        let height = 800u32;

        let scale = self.camera.scales(width as f64, height as f64);
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
        let body_vertex_count = self.body_scratch.len() as u32;
        let trail_bodies = self.trails.len() as u64;

        let (line_pipeline, body_pipeline, trail_vertex_buffer, body_vertex_buffer, color_format) = {
            let rcx = self.rcx.as_ref().unwrap();
            (
                rcx.line_pipeline.clone(),
                rcx.body_pipeline.clone(),
                rcx.trail_vertex_buffer.clone(),
                rcx.body_vertex_buffer.clone(),
                rcx.swapchain.image_format(),
            )
        };
        let line_layout = line_pipeline.layout().clone();
        let body_layout = body_pipeline.layout().clone();

        {
            if let Some(rcx) = self.rcx.as_mut() {
                if let Some(previous) = rcx.previous_frame_end.take() {
                    drop(previous);
                }
            }
        }

        {
            let mut guard = body_vertex_buffer.write().unwrap();
            guard.copy_from_slice(&self.body_scratch);
        }
        {
            let mut guard = trail_vertex_buffer.write().unwrap();
            guard.copy_from_slice(&self.trail_scratch);
        }

        let image = Image::new(
            &self.memory_allocator,
            &ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: color_format,
                extent: [width, height, 1],
                usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();
        let view = ImageView::new_default(&image).unwrap();

        let readback: Subbuffer<[u8]> = Buffer::from_iter(
            &self.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            std::iter::repeat(0u8).take((width * height * 4) as usize),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.012, 0.012, 0.03, 1.0].into()),
                    ..RenderingAttachmentInfo::new(view)
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(
                0,
                [Viewport {
                    offset: [0.0, 0.0],
                    extent: [width as f32, height as f32],
                    min_depth: 0.0,
                    max_depth: 1.0,
                }]
                .into_iter()
                .collect(),
            )
            .unwrap()
            .bind_pipeline_graphics(line_pipeline)
            .unwrap()
            .push_constants(line_layout, 0, push)
            .unwrap();

        for i in 0..trail_bodies {
            let start = i * TRAIL_LEN as u64;
            let slice = trail_vertex_buffer
                .clone()
                .slice(start..start + TRAIL_LEN as u64);
            builder.bind_vertex_buffers(0, slice).unwrap();
            unsafe { builder.draw(TRAIL_LEN as u32, 1, 0, 0) }.unwrap();
        }

        builder
            .bind_pipeline_graphics(body_pipeline)
            .unwrap()
            .push_constants(body_layout, 0, push)
            .unwrap()
            .bind_vertex_buffers(0, body_vertex_buffer)
            .unwrap();
        unsafe { builder.draw(body_vertex_count, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();
        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::new(image, readback.clone()))
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap)
            .unwrap();
        future.wait(None).map_err(Validated::unwrap).unwrap();

        let data = readback.read().unwrap();
        let bgra = matches!(
            color_format,
            Format::B8G8R8A8_UNORM | Format::B8G8R8A8_SRGB | Format::B8G8R8A8_SNORM
        );
        let mut ppm = format!("P6\n{width} {height}\n255\n").into_bytes();
        for px in data.chunks_exact(4) {
            if bgra {
                ppm.extend_from_slice(&[px[2], px[1], px[0]]);
            } else {
                ppm.extend_from_slice(&[px[0], px[1], px[2]]);
            }
        }
        std::fs::write(path, ppm).unwrap();
        println!("debug frame written to {path}");
    }
}

fn main() -> Result<(), impl Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
