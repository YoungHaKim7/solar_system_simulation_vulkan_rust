pub(crate) const START_VIEW_HEIGHT: f64 = 70.0;

pub(crate) struct Camera {
    pub(crate) center: [f64; 2],
    pub(crate) height: f64,
}

impl Camera {
    pub(crate) fn scales(&self, width: f64, height_px: f64) -> [f64; 2] {
        let aspect = width / height_px;
        [2.0 / (self.height * aspect), 2.0 / self.height]
    }

    pub(crate) fn offsets(&self, s: [f64; 2]) -> [f64; 2] {
        [-self.center[0] * s[0], -self.center[1] * s[1]]
    }

    pub(crate) fn screen_to_world(&self, px: f64, py: f64, width: f64, height_px: f64) -> [f64; 2] {
        let s = self.scales(width, height_px);
        let o = self.offsets(s);
        let nx = px / width * 2.0 - 1.0;
        let ny = py / height_px * 2.0 - 1.0;
        [(nx - o[0]) / s[0], (-ny - o[1]) / s[1]]
    }

    pub(crate) fn zoom(&mut self, factor: f64, px: f64, py: f64, width: f64, height_px: f64) {
        let before = self.screen_to_world(px, py, width, height_px);
        self.height = (self.height / factor).clamp(0.15, 600.0);
        let after = self.screen_to_world(px, py, width, height_px);
        self.center[0] += before[0] - after[0];
        self.center[1] += before[1] - after[1];
    }

    pub(crate) fn pan(&mut self, dx_px: f64, dy_px: f64, height_px: f64) {
        let world_per_px = self.height / height_px;
        self.center[0] -= dx_px * world_per_px;
        self.center[1] += dy_px * world_per_px;
    }
}
