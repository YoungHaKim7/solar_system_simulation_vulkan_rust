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
