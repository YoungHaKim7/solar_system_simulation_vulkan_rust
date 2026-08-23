use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer, RenderingAttachmentInfo, RenderingInfo,
    },
    device::Device,
    format::Format,
    image::{ImageUsage, view::ImageView},
    instance::Instance,
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
            vertex_input::{Vertex, VertexDefinition, VertexInputState},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    shader::EntryPoint,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{self, GpuFuture},
};
use winit::{dpi::LogicalSize, event_loop::ActiveEventLoop, window::Window};

use crate::{
    shaders::{body_fs, body_vs, line_fs, line_vs},
    trails::TRAIL_LEN,
    vertices::{BodyVertex, PushConstants, TrailVertex},
};

pub(crate) struct RenderContext {
    pub(crate) window: Arc<Window>,
    pub(crate) swapchain: Arc<Swapchain>,
    pub(crate) attachment_image_views: Vec<Arc<ImageView>>,
    body_pipeline: Arc<GraphicsPipeline>,
    line_pipeline: Arc<GraphicsPipeline>,
    pub(crate) viewport: Viewport,
    pub(crate) body_vertex_buffer: Subbuffer<[BodyVertex]>,
    pub(crate) trail_vertex_buffer: Subbuffer<[TrailVertex]>,
    pub(crate) recreate_swapchain: bool,
    pub(crate) previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl RenderContext {
    pub(crate) fn new(
        event_loop: &ActiveEventLoop,
        instance: &Arc<Instance>,
        device: &Arc<Device>,
        memory_allocator: &Arc<StandardMemoryAllocator>,
        body_vertices: Vec<BodyVertex>,
        trail_vertices: Vec<TrailVertex>,
    ) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes().with_inner_size(LogicalSize::new(1280.0, 800.0)),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();

            let (image_format, _) = device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            Swapchain::new(
                device,
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

        let body_vs_entry = unsafe { body_vs::load(device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let body_fs_entry = unsafe { body_fs::load(device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let body_vertex_input = BodyVertex::per_vertex().definition(&body_vs_entry).unwrap();
        let body_pipeline = create_graphics_pipeline(
            device,
            &body_vs_entry,
            &body_fs_entry,
            body_vertex_input,
            PrimitiveTopology::TriangleList,
            swapchain.image_format(),
        );

        let line_vs_entry = unsafe { line_vs::load(device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let line_fs_entry = unsafe { line_fs::load(device) }
            .unwrap()
            .entry_point("main")
            .unwrap();
        let line_vertex_input = TrailVertex::per_vertex()
            .definition(&line_vs_entry)
            .unwrap();
        let line_pipeline = create_graphics_pipeline(
            device,
            &line_vs_entry,
            &line_fs_entry,
            line_vertex_input,
            PrimitiveTopology::LineStrip,
            swapchain.image_format(),
        );

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let body_vertex_buffer = Buffer::from_iter(
            memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            body_vertices,
        )
        .unwrap();

        let trail_vertex_buffer = Buffer::from_iter(
            memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            trail_vertices,
        )
        .unwrap();

        Self {
            window,
            swapchain,
            attachment_image_views,
            body_pipeline,
            line_pipeline,
            viewport,
            body_vertex_buffer,
            trail_vertex_buffer,
            recreate_swapchain: false,
            previous_frame_end: Some(sync::now(device.clone()).boxed()),
        }
    }

    pub(crate) fn refresh_swapchain(&mut self, extent: [u32; 2]) {
        let (new_swapchain, new_images) = self
            .swapchain
            .recreate(&SwapchainCreateInfo {
                image_extent: extent,
                ..self.swapchain.create_info()
            })
            .expect("failed to recreate swapchain");

        self.swapchain = new_swapchain;
        self.attachment_image_views = new_images
            .iter()
            .map(|image| ImageView::new_default(image).unwrap())
            .collect::<Vec<_>>();
        self.viewport.extent = extent.map(|v| v as f32);
        self.recreate_swapchain = false;
    }

    pub(crate) fn record_scene(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        attachment: Arc<ImageView>,
        viewport: Viewport,
        push: PushConstants,
        trail_strips: u64,
        body_vertex_count: u32,
    ) {
        builder
            .begin_rendering(RenderingInfo {
                color_attachments: vec![Some(RenderingAttachmentInfo {
                    load_op: AttachmentLoadOp::Clear,
                    store_op: AttachmentStoreOp::Store,
                    clear_value: Some([0.012, 0.012, 0.03, 1.0].into()),
                    ..RenderingAttachmentInfo::new(attachment)
                })],
                ..Default::default()
            })
            .unwrap()
            .set_viewport(0, [viewport].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.line_pipeline.clone())
            .unwrap()
            .push_constants(self.line_pipeline.layout().clone(), 0, push)
            .unwrap();

        for i in 0..trail_strips {
            let start = i * TRAIL_LEN as u64;
            let slice = self
                .trail_vertex_buffer
                .clone()
                .slice(start..start + TRAIL_LEN as u64);
            builder.bind_vertex_buffers(0, slice).unwrap();
            unsafe { builder.draw(TRAIL_LEN as u32, 1, 0, 0) }.unwrap();
        }

        builder
            .bind_pipeline_graphics(self.body_pipeline.clone())
            .unwrap()
            .push_constants(self.body_pipeline.layout().clone(), 0, push)
            .unwrap()
            .bind_vertex_buffers(0, self.body_vertex_buffer.clone())
            .unwrap();

        unsafe { builder.draw(body_vertex_count, 1, 0, 0) }.unwrap();

        builder.end_rendering().unwrap();
    }
}

fn create_graphics_pipeline(
    device: &Arc<Device>,
    vs: &EntryPoint,
    fs: &EntryPoint,
    vertex_input_state: VertexInputState,
    topology: PrimitiveTopology,
    color_format: Format,
) -> Arc<GraphicsPipeline> {
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::from_stages(device, &stages).unwrap();

    let subpass = PipelineRenderingCreateInfo {
        color_attachment_formats: &[Some(color_format)],
        ..Default::default()
    };

    GraphicsPipeline::new(
        device,
        None,
        &GraphicsPipelineCreateInfo {
            stages: &stages,
            vertex_input_state: Some(&vertex_input_state),
            input_assembly_state: Some(&InputAssemblyState {
                topology,
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
}
