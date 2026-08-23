use vulkano::{
    Validated,
    format::Format,
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo},
    image::{Image, ImageCreateInfo, ImageType, ImageUsage, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::viewport::Viewport,
    sync::{self, GpuFuture},
};

use crate::{
    app::App,
    vertices::{PushConstants, fill_body_vertices},
};

impl App {
    pub(crate) fn debug_dump_frame(&mut self, path: &str) {
        self.sim.advance(365.25 * 3.0);
        self.trails.sample(self.sim.t_days, &self.sim.bodies);
        fill_body_vertices(&self.sim.bodies, &mut self.body_scratch);
        let trail_strips = self.trails.len() as u64;
        let trail_vertices = self.trails.flatten();

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

        let rcx = self.rcx.as_mut().unwrap();
        let color_format = rcx.swapchain.image_format();

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

        let image = Image::new(
            &self.gpu.memory_allocator,
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
            &self.gpu.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            std::iter::repeat_n(0u8, (width * height * 4) as usize),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        rcx.record_scene(
            &mut builder,
            view,
            Viewport {
                offset: [0.0, 0.0],
                extent: [width as f32, height as f32],
                min_depth: 0.0,
                max_depth: 1.0,
            },
            push,
            trail_strips,
            body_vertex_count,
        );

        builder
            .copy_image_to_buffer(CopyImageToBufferInfo::new(image, readback.clone()))
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let future = sync::now(self.gpu.device.clone())
            .then_execute(self.gpu.queue.clone(), command_buffer)
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
        for px in data.as_chunks::<4>().0 {
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
