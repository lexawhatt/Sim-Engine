//! Sim;Engine is a rendering layer for simulation products.
//!
//! It intentionally does not model physics, biology, chemistry, math domains,
//! constants, entities, or plugins. Those belong to Sim;X. This crate accepts
//! already-computed visual state and turns it into a smooth, styled 2D scene.

#![warn(missing_docs)]

mod camera;
mod color;
mod easing;
mod math;
mod scene;
mod tween;

#[cfg(feature = "wgpu")]
mod wgpu_renderer;

pub use camera::{Camera2d, Projection2d, Viewport};
pub use color::{Color, Palette};
pub use easing::Easing;
pub use math::{Rect, Vec2};
pub use scene::{
    Circle, DrawCommand, Line, Polyline, RectShape, Scene, Shadow, ShapeStyle, Stroke,
};
pub use tween::{Interpolate, Tween};

#[cfg(feature = "wgpu")]
pub use wgpu_renderer::{RenderStatus, RendererInitError, RendererSurfaceStatus, WgpuRenderer};
