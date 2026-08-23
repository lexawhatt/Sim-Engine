//! Sim;Engine is a rendering layer for simulation products.
//!
//! It intentionally does not model physics, biology, chemistry, math domains,
//! constants, entities, or plugins. Those belong to the host application's
//! domain layer. This crate accepts already-computed visual state and turns it
//! into a smooth, styled 2D scene.

#![warn(missing_docs)]

mod camera;
mod color;
mod easing;
mod field;
mod math;
mod scene;
mod tween;

#[cfg(feature = "wgpu")]
mod renderer;

pub use camera::{
    Camera2d, Camera2dError, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportError, PhysicalScreenPosition, Projection2d, Projection2dError,
};
pub use color::{Color, Palette};
pub use easing::Easing;
pub use field::{ColorMap, ColorMapError, ColorStop, ScalarField, ScalarFieldError};
pub use math::{Rect, Vec2};
pub use scene::{
    Circle, DrawCommand, Fill, Layer, Line, LinearGradient, Polyline, RadialGradient, RectShape,
    Scene, SceneCommand, SceneError, ScenePrimitive, ScreenClipRect, Shadow, ShapeStyle, Stroke,
};
pub use tween::{Interpolate, Tween, TweenError};

#[cfg(feature = "wgpu")]
pub use renderer::{
    BlendMode, DynamicMesh2d, DynamicMeshError, DynamicMeshRenderError, DynamicMeshUpdateReport,
    DynamicVertex2d, LayeredVisualizationError, LayeredVisualizationOptions, ParticleBudgetError,
    ParticleField2d, ParticleFieldError, ParticleFieldRenderError, ParticleFieldUpdateReport,
    ParticleInstance2d, ParticleInstanceError, ParticleRenderBudget, ParticleStatistics,
    PreparedScene, PreparedSceneError, PreparedSceneRenderError, RenderReport, RenderStatus,
    RenderTarget2d, RenderTargetError, RenderTargetLoad, RendererConfigurationError,
    RendererCoordinateError, RendererFrameError, RendererFrameMetrics, RendererInitError,
    RendererPresentMode, RendererSurfaceStatus, ScalarFieldRenderError, ScalarFieldSampling,
    ScalarFieldTexture, ScalarFieldTextureError, ScalarFieldUploadReport, TessellationStats,
    TrailBuffer2d, WgpuRenderer, WgpuRendererOptions,
};
