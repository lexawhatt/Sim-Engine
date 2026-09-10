#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

mod camera;
mod color;
mod easing;
mod field;
mod math;
mod mesh3d;
mod particle;
mod pseudo3d;
mod scene;
mod screen;
mod tween;
mod units;

#[cfg(test)]
mod test_allocations;

#[cfg(feature = "wgpu")]
mod renderer;

pub use camera::{
    Camera2d, Camera2dError, LogicalScreenPosition, LogicalScreenVector, LogicalViewport,
    LogicalViewportError, LogicalViewportRegion, PhysicalScreenPosition, Projection2d,
    Projection2dError,
};
pub use color::{Color, Palette};
pub use easing::Easing;
pub use field::{ColorMap, ColorMapError, ColorStop, ScalarField, ScalarFieldError};
pub use math::{Rect, Vec2};
pub use mesh3d::{
    Mesh3d, Mesh3dError, Mesh3dStyleError, MeshEdge3d, MeshStyle3d, SurfaceStyle3d,
    TextureCoordinate2d, WireframeStyle3d,
};
pub use particle::{ParticleInstance2d, ParticleInstanceError};
pub use pseudo3d::{
    Camera3d, ProjectedPoint3d, Projection3d, Pseudo3dError, Rotation3d, Transform3d, Vec3,
};
pub use scene::{
    Circle, DrawCommand, Fill, Layer, Line, LinearGradient, MAX_STROKE_DASH_SUBSEGMENTS, Polyline,
    PrimitiveCommandCounts, RadialGradient, RectShape, Scene, SceneBudget, SceneBudgetResource,
    SceneCommand, SceneError, ScenePrimitive, SceneStatistics, ScreenClipRect, Shadow, ShapeStyle,
    Stroke, StrokeCap2d, StrokeDashPattern2d, StrokeJoin2d, StrokeMarker2d, StrokeMarkerAnchor2d,
    StrokeStyle2d, StrokeStyleError, StrokeWidthMode2d,
};
pub use screen::ScreenScene;
pub use tween::{Interpolate, Tween, TweenError};
pub use units::{LogicalPixels, PhysicalPerLogical, UnitError, WorldLength};

#[cfg(feature = "wgpu")]
pub use renderer::{
    BlendMode, DynamicMesh2d, DynamicMeshBudget, DynamicMeshBudgetResource, DynamicMeshError,
    DynamicMeshRenderError, DynamicMeshUpdateReport, DynamicVertex2d, FrameBudget,
    FrameBudgetResource, FrameCacheBudget, FrameCacheStatistics, FrameComposer, FrameComposerError,
    FramePassOptions, FrameReport, FrameSourceKind, FrameSourceStatistics, FrameStatistics,
    GlyphAtlas2d, GlyphAtlasBudget, GlyphAtlasEntry, GlyphError, GlyphId, GlyphRun2d,
    GlyphRunBounds, GlyphRunBudget, GlyphRunStatistics, GlyphRunUploadReport, GlyphUploadReport,
    Image2d, ImageBatch2d, ImageBatchBudget, ImageBatchPlacement, ImageBatchUploadReport,
    ImageBudget, ImageError, ImageSampling, ImageSprite2d, ImageTexelRect, ImageUploadReport,
    LayeredVisualizationError, LayeredVisualizationOptions, LayeredVisualizationReport,
    Mesh3dInstance, Mesh3dObjectError, Mesh3dPreflightReport, Mesh3dRenderBudget,
    Mesh3dRenderError, Mesh3dRenderReport, Mesh3dResourceError, Mesh3dUploadBudget,
    Mesh3dUploadBudgetResource, Mesh3dUploadReport, Object3dId, ParticleBudgetError,
    ParticleField2d, ParticleFieldError, ParticleFieldRenderError, ParticleFieldUpdateReport,
    ParticleRenderBudget, ParticleStatistics, PositionedGlyph2d, PreparedScene, PreparedSceneError,
    PreparedSceneRenderError, PreparedScreenScene, RenderReport, RenderStatus, RenderTarget2d,
    RenderTarget3d, RenderTargetError, RenderTargetLoad, RendererConfigurationError,
    RendererCoordinateError, RendererFrameError, RendererFrameMetrics, RendererInitError,
    RendererPresentMode, RendererSurfacePresentMode, RendererSurfaceStatus, RetainedMesh3d,
    ScalarFieldRenderError, ScalarFieldSampling, ScalarFieldTexture, ScalarFieldTextureError,
    ScalarFieldUploadReport, Scene3d, Scene3dBudget, Scene3dBudgetResource, Scene3dError,
    Scene3dMeshUpdateReport, Scene3dRestoreReport, Scene3dStatistics, TessellationStats, Texture3d,
    Texture3dError, TextureMaterial3d, TrailBuffer2d, WgpuRenderer, WgpuRendererOptions,
};
