# opencode + claude(glm-5.3) AI(260823)

- Done. The 1298-line `main.rs` is now 11 modules, each owning one responsibility:

| File            | Lines | Function                                                                   |
| --------------- | ----- | -------------------------------------------------------------------------- |
| `main.rs`       | 22    | Module wiring + `main()`                                                   |
| `shaders.rs`    | 27    | 4 SPIR-V shader wrappers                                                   |
| `camera.rs`     | 39    | Pan/zoom, screen↔world math                                                |
| `vertices.rs`   | 63    | `BodyVertex`/`TrailVertex`/`PushConstants`, quad geometry, vertex building |
| `trails.rs`     | 83    | Trail sampling + fade (`Trails`)                                           |
| `simulation.rs` | 105   | N-body physics, velocity-Verlet integrator                                 |
| `bodies.rs`     | 106   | Body catalog (`BODIES`) + orbital periods                                  |
| `gpu.rs`        | 116   | Vulkan bootstrap: instance, device selection, queues, allocators           |
| `debug.rs`      | 140   | Offscreen PPM dump (`SOLAR_DUMP_FRAME`)                                    |
| `renderer.rs`   | 318   | Swapchain, pipelines, frame recording                                      |
| `app.rs`        | 355   | App state, winit events, input, frame loop                                 |

**Two small dedups along the way** (the clearest divide-and-conquer wins):
- `renderer.rs` — the two ~55-line copy-pasted pipeline blocks became one `create_graphics_pipeline()` factory (topology is the only real difference).
- `RenderContext::record_scene()` — the trail-then-body draw sequence, duplicated between `redraw` and `debug_dump_frame`, now exists once.

**Verification** (behavior must be unchanged):
- `cargo build` passes clean.
- Ran before/after with `SOLAR_DUMP_FRAME=1` and compared the 1280×800 output: 1,024,000 pixels, 18 differ by max Δ30 — but two runs of the *same unmodified binary* differ by 23 pixels / Δ59 on the *same rows*. So the residual is GPU rasterization nondeterminism; the scene data itself is bit-identical (identical stdout periods too).
- The 7 clippy warnings are pre-existing style lints in verbatim-moved code (collapsible `if`s, `repeat().take()`) — I deliberately didn't fold them in, to keep this a pure structural change. Happy to clean them up separately if you want.

Changes are uncommitted, so `git diff` / `git checkout -- src/` reviews or reverts cleanly.

# clippy

- All 7 clippy errors fixed, `cargo clippy --all-features --all-targets -- -D warnings` now passes clean:

- **`src/app.rs:150`** — collapsed `if self.dragging` + `if let Some(prev)` into one let-chain (edition 2024)
- **`src/app.rs:171`** — dropped the redundant `as f64` on `p.y` (already `f64` in winit's `PhysicalPosition`)
- **`src/app.rs:173`** — collapsed the three nested ifs (`factor` check + two `if let`s) into a single let-chain
- **`src/debug.rs:86`** — `repeat(0u8).take(n)` → `repeat_n(0u8, n)`
- **`src/debug.rs:130`** — `chunks_exact(4)` → `as_chunks::<4>().0`; I verified the return type first (`(&[[u8; 4]], &[u8])` — chunks come first), so `.0` is the chunks, not the remainder
- **`src/trails.rs:57`** — range loop over queue indices → `bodies.iter().enumerate().take(self.queues.len()).skip(1)`, preserving the exact original bounds (`1..queues.len()`); `px[0]`/`px[1]`/`px[2]` indexing still works since chunks are now `&[u8; 4]`
