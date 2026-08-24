#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod camera;
mod color;
mod easing;
mod field;
mod math;
mod pseudo3d;
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
pub use pseudo3d::{
    Camera3d, ProjectedPoint3d, Projection3d, Pseudo3dError, Rotation3d, Transform3d, Vec3,
};
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
