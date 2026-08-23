pub(crate) mod body_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "../assets/body.vert",
    }
}

pub(crate) mod body_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "../assets/body.frag",
    }
}

pub(crate) mod line_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "../assets/line.vert",
    }
}

pub(crate) mod line_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "../assets/line.frag"
    }
}
