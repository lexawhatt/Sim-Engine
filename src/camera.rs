use crate::{
    math::Vec2,
    tween::{Interpolate, Tween},
};

/// Pixel dimensions of the render target currently being drawn into.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    /// Width in physical screen pixels.
    pub width: f32,
    /// Height in physical screen pixels.
    pub height: f32,
}

impl Viewport {
    /// Builds a viewport from physical pixel dimensions.
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Returns the center point in screen pixel coordinates.
    pub fn center(self) -> Vec2 {
        Vec2::new(self.width * 0.5, self.height * 0.5)
    }

    /// Returns viewport size in screen pixels.
    pub fn size(self) -> Vec2 {
        Vec2::new(self.width, self.height)
    }
}

/// Lightweight pseudo-depth projection for 2D scenes.
///
/// This is not a mesh renderer or z-buffer. It offsets 2D points using a scalar
/// depth value so Sim;X can create camera-like depth transitions without leaving
/// the 2D rendering model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection2d {
    /// Projection tilt in radians.
    pub tilt: f32,
    /// Multiplier applied to caller-provided depth values.
    pub depth_scale: f32,
}

impl Projection2d {
    /// Projection that leaves points in their original 2D plane.
    pub const FLAT: Self = Self {
        tilt: 0.0,
        depth_scale: 1.0,
    };

    /// Builds a projection from tilt in radians and depth scale.
    pub fn new(tilt: f32, depth_scale: f32) -> Self {
        Self { tilt, depth_scale }
    }

    /// Projects a world-space point with caller-provided pseudo-depth.
    pub fn project(self, point: Vec2, depth: f32) -> Vec2 {
        let lifted = depth * self.depth_scale;
        Vec2::new(
            point.x + lifted * self.tilt.sin() * 0.5,
            point.y * self.tilt.cos() + lifted * self.tilt.sin(),
        )
    }
}

/// 2D camera mapping simulation world coordinates into screen pixels.
///
/// `center` and all input positions are in world units. `zoom` is pixels per
/// world unit. `rotation` is in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2d {
    /// World-space point displayed at the viewport center.
    pub center: Vec2,
    /// Scale factor in screen pixels per world unit.
    pub zoom: f32,
    /// Clockwise screen rotation in radians after projection.
    pub rotation: f32,
    /// Optional pseudo-depth projection applied before camera transform.
    pub projection: Projection2d,
}

impl Camera2d {
    /// Builds a camera centered on a world-space point.
    ///
    /// Non-positive zoom values are accepted here but clamped by conversion
    /// methods that need division.
    pub fn new(center: Vec2, zoom: f32) -> Self {
        Self {
            center,
            zoom,
            rotation: 0.0,
            projection: Projection2d::FLAT,
        }
    }

    /// Converts a world-space point into screen pixel coordinates.
    pub fn world_to_screen(self, world: Vec2, viewport: Viewport) -> Vec2 {
        self.projected_world_to_screen(world, 0.0, viewport)
    }

    /// Converts a world-space point and pseudo-depth into screen pixels.
    pub fn projected_world_to_screen(self, world: Vec2, depth: f32, viewport: Viewport) -> Vec2 {
        let projected = self.projection.project(world, depth);
        let translated = projected - self.center;
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let rotated = Vec2::new(
            translated.x * cos - translated.y * sin,
            translated.x * sin + translated.y * cos,
        );

        Vec2::new(
            viewport.width * 0.5 + rotated.x * self.zoom,
            viewport.height * 0.5 - rotated.y * self.zoom,
        )
    }

    /// Converts screen pixel coordinates back into world coordinates.
    ///
    /// Pseudo-depth projection cannot be reversed by this function. It assumes
    /// the flat `depth = 0.0` plane.
    pub fn screen_to_world(self, screen: Vec2, viewport: Viewport) -> Vec2 {
        let zoom = self.zoom.max(0.0001);
        let translated = Vec2::new(
            (screen.x - viewport.width * 0.5) / zoom,
            -(screen.y - viewport.height * 0.5) / zoom,
        );
        let cos = self.rotation.cos();
        let sin = self.rotation.sin();
        let rotated = Vec2::new(
            translated.x * cos + translated.y * sin,
            -translated.x * sin + translated.y * cos,
        );

        rotated + self.center
    }

    /// Creates a tween initialized with this camera state.
    pub fn tween(self) -> Tween<Self> {
        Tween::new(self)
    }
}

impl Default for Camera2d {
    fn default() -> Self {
        Self::new(Vec2::ZERO, 1.0)
    }
}

impl Interpolate for Projection2d {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        Self {
            tilt: self.tilt.interpolate(end.tilt, amount),
            depth_scale: self.depth_scale.interpolate(end.depth_scale, amount),
        }
    }
}

impl Interpolate for Camera2d {
    fn interpolate(self, end: Self, amount: f32) -> Self {
        Self {
            center: self.center.interpolate(end.center, amount),
            zoom: self.zoom.interpolate(end.zoom, amount).max(0.0001),
            rotation: self.rotation.interpolate(end.rotation, amount),
            projection: self.projection.interpolate(end.projection, amount),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Camera2d, Projection2d, Vec2, Viewport};

    #[test]
    fn camera_roundtrip_without_projection() {
        let camera = Camera2d {
            center: Vec2::new(20.0, -10.0),
            zoom: 2.5,
            rotation: 0.35,
            projection: Projection2d::FLAT,
        };
        let viewport = Viewport::new(1280.0, 720.0);
        let world = Vec2::new(44.0, 12.0);

        let screen = camera.world_to_screen(world, viewport);
        let roundtrip = camera.screen_to_world(screen, viewport);

        assert!((roundtrip.x - world.x).abs() < 0.001);
        assert!((roundtrip.y - world.y).abs() < 0.001);
    }
}
