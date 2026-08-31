use std::{
    error::Error,
    fmt,
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
};

use crate::{
    Color, Interpolate, LogicalPixels, LogicalScreenPosition, LogicalScreenVector, Rect, Vec2,
    WorldLength,
};

pub(crate) const CIRCLE_SEGMENTS: usize = 64;
pub(crate) const ROUND_CAP_SEGMENTS: usize = 16;
pub(crate) const CORNER_SEGMENTS: usize = 12;
pub(crate) const TESSELLATED_VERTEX_BYTES: usize = 20 * size_of::<f32>();
const MAX_DASH_ELEMENTS: usize = 8;
const MAX_MITER_LIMIT: f32 = 1_000.0;
const MIN_STROKE_TURN_SINE: f64 = 0.000_001;
/// Maximum visible dash pieces one command may request from tessellation.
pub const MAX_STROKE_DASH_SUBSEGMENTS: usize = 1_000_000;

/// Primitive category attached to structured scene validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePrimitive {
    /// Circle geometry or style.
    Circle,
    /// Rectangle geometry or style.
    Rect,
    /// Single line geometry or stroke.
    Line,
    /// Connected polyline geometry or stroke.
    Polyline,
}

/// Work category constrained by a [`SceneBudget`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SceneBudgetResource {
    /// Accepted scene commands.
    Commands,
    /// Points retained by polyline commands.
    Points,
    /// Conservative upper bound for generated triangle-list vertices.
    TessellatedVertices,
    /// Command payload bytes retained by the scene.
    RetainedBytes,
    /// Bytes uploaded for generated scene vertices.
    UploadBytes,
    /// Draw batches caused by ordering and clip changes.
    DrawBatches,
}

/// Explicit upper bounds for engine-owned ordinary-scene work.
///
/// Limits may be zero to construct a background-only scene. Retained bytes
/// count command values and owned polyline storage, excluding allocator
/// bookkeeping. Vertex and upload limits use the renderer's conservative
/// triangle-list estimate, which is checked again after tessellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneBudget {
    max_commands: usize,
    max_points: usize,
    max_tessellated_vertices: usize,
    max_retained_bytes: usize,
    max_upload_bytes: usize,
    max_draw_batches: usize,
}

impl SceneBudget {
    /// Builds a complete budget for ordinary scene construction and rendering.
    pub const fn new(
        max_commands: usize,
        max_points: usize,
        max_tessellated_vertices: usize,
        max_retained_bytes: usize,
        max_upload_bytes: usize,
        max_draw_batches: usize,
    ) -> Self {
        Self {
            max_commands,
            max_points,
            max_tessellated_vertices,
            max_retained_bytes,
            max_upload_bytes,
            max_draw_batches,
        }
    }

    /// Returns the maximum accepted command count.
    pub const fn max_commands(self) -> usize {
        self.max_commands
    }

    /// Returns the maximum retained polyline point count.
    pub const fn max_points(self) -> usize {
        self.max_points
    }

    /// Returns the maximum conservatively estimated vertex count.
    pub const fn max_tessellated_vertices(self) -> usize {
        self.max_tessellated_vertices
    }

    /// Returns the maximum retained command payload bytes.
    pub const fn max_retained_bytes(self) -> usize {
        self.max_retained_bytes
    }

    /// Returns the maximum generated vertex upload bytes.
    pub const fn max_upload_bytes(self) -> usize {
        self.max_upload_bytes
    }

    /// Returns the maximum draw-batch count.
    pub const fn max_draw_batches(self) -> usize {
        self.max_draw_batches
    }
}

/// Construction and conservative rendering work accumulated by a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SceneStatistics {
    requested_commands: usize,
    accepted_commands: usize,
    rejected_commands: usize,
    requested_by_primitive: PrimitiveCommandCounts,
    accepted_by_primitive: PrimitiveCommandCounts,
    rejected_by_primitive: PrimitiveCommandCounts,
    retained_points: usize,
    estimated_tessellated_vertices: usize,
    retained_bytes: usize,
    estimated_upload_bytes: usize,
    estimated_draw_batches: usize,
}

impl SceneStatistics {
    /// Returns all insertion attempts, including invalid and over-budget work.
    pub const fn requested_commands(self) -> usize {
        self.requested_commands
    }

    /// Returns commands currently retained by the scene.
    pub const fn accepted_commands(self) -> usize {
        self.accepted_commands
    }

    /// Returns insertion attempts rejected by validation, budget, or allocation.
    pub const fn rejected_commands(self) -> usize {
        self.rejected_commands
    }

    /// Returns insertion attempts grouped by primitive category.
    pub const fn requested_by_primitive(self) -> PrimitiveCommandCounts {
        self.requested_by_primitive
    }

    /// Returns currently retained commands grouped by primitive category.
    pub const fn accepted_by_primitive(self) -> PrimitiveCommandCounts {
        self.accepted_by_primitive
    }

    /// Returns rejected insertion attempts grouped by primitive category.
    pub const fn rejected_by_primitive(self) -> PrimitiveCommandCounts {
        self.rejected_by_primitive
    }

    /// Returns points retained by accepted polyline commands.
    pub const fn retained_points(self) -> usize {
        self.retained_points
    }

    /// Returns the conservative upper bound for tessellated vertices.
    pub const fn estimated_tessellated_vertices(self) -> usize {
        self.estimated_tessellated_vertices
    }

    /// Returns tracked command and owned polyline payload bytes.
    pub const fn retained_bytes(self) -> usize {
        self.retained_bytes
    }

    /// Returns estimated bytes uploaded for generated vertices.
    pub const fn estimated_upload_bytes(self) -> usize {
        self.estimated_upload_bytes
    }

    /// Returns the conservative draw-batch upper bound.
    pub const fn estimated_draw_batches(self) -> usize {
        self.estimated_draw_batches
    }

    fn with_command(mut self, command: &DrawCommand) -> Self {
        let vertices = command.estimated_tessellated_vertices();
        self.accepted_commands = self.accepted_commands.saturating_add(1);
        self.accepted_by_primitive.increment(command.primitive());
        self.retained_points = self
            .retained_points
            .saturating_add(command.retained_point_count());
        self.estimated_tessellated_vertices =
            self.estimated_tessellated_vertices.saturating_add(vertices);
        self.retained_bytes = self.retained_bytes.saturating_add(command.retained_bytes());
        self.estimated_upload_bytes = self
            .estimated_upload_bytes
            .saturating_add(vertices.saturating_mul(TESSELLATED_VERTEX_BYTES));
        self.estimated_draw_batches = self.estimated_draw_batches.saturating_add(1);
        self
    }
}

/// Command counters grouped by ordinary 2D primitive category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrimitiveCommandCounts {
    circles: usize,
    rectangles: usize,
    lines: usize,
    polylines: usize,
}

impl PrimitiveCommandCounts {
    /// Returns circle command count.
    pub const fn circles(self) -> usize {
        self.circles
    }

    /// Returns rectangle command count.
    pub const fn rectangles(self) -> usize {
        self.rectangles
    }

    /// Returns single-line command count.
    pub const fn lines(self) -> usize {
        self.lines
    }

    /// Returns connected-polyline command count.
    pub const fn polylines(self) -> usize {
        self.polylines
    }

    /// Returns all primitive commands represented by these counters.
    pub const fn total(self) -> usize {
        self.circles
            .saturating_add(self.rectangles)
            .saturating_add(self.lines)
            .saturating_add(self.polylines)
    }

    pub(crate) fn increment(&mut self, primitive: ScenePrimitive) {
        match primitive {
            ScenePrimitive::Circle => self.circles = self.circles.saturating_add(1),
            ScenePrimitive::Rect => self.rectangles = self.rectangles.saturating_add(1),
            ScenePrimitive::Line => self.lines = self.lines.saturating_add(1),
            ScenePrimitive::Polyline => self.polylines = self.polylines.saturating_add(1),
        }
    }

    pub(crate) fn adding(mut self, other: Self) -> Self {
        self.circles = self.circles.saturating_add(other.circles);
        self.rectangles = self.rectangles.saturating_add(other.rectangles);
        self.lines = self.lines.saturating_add(other.lines);
        self.polylines = self.polylines.saturating_add(other.polylines);
        self
    }
}

/// Reason a command or temporary scene state was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneError {
    /// Clear color is non-finite or outside normalized RGBA.
    InvalidBackground,
    /// Primitive coordinates contain NaN or infinity.
    NonFiniteGeometry(ScenePrimitive),
    /// Radius, size, corner radius, or stroke width is outside its valid range.
    InvalidDimension(ScenePrimitive),
    /// A line segment, or at least one consecutive polyline segment, is not drawable.
    DegenerateGeometry(ScenePrimitive),
    /// Consecutive polyline segments form an exact or numerically indistinguishable reversal.
    ///
    /// A retraced centerline has no interior-disjoint alpha-blended stroke
    /// representation. `vertex_index` identifies the reversing path point.
    DegenerateStrokeTurn {
        /// Primitive containing the reversal.
        primitive: ScenePrimitive,
        /// Zero-based index of the shared reversing point.
        vertex_index: usize,
    },
    /// Shape has no fill, stroke, or shadow.
    MissingStyle(ScenePrimitive),
    /// Fill color or gradient configuration is invalid.
    InvalidFill(ScenePrimitive),
    /// Stroke width or color is invalid.
    InvalidStroke(ScenePrimitive),
    /// A dash pattern would expand into more retained triangle pieces than its
    /// explicit per-command limit.
    StrokeExpansionLimitExceeded {
        /// Primitive whose stroke exceeded the limit.
        primitive: ScenePrimitive,
        /// Configured maximum number of emitted visible subsegments.
        limit: usize,
        /// Minimum visible-subsegment count that proved the limit was exceeded.
        required: usize,
    },
    /// Dash/gap boundaries collapse at the source path's `f32` coordinate scale.
    UnrepresentableStrokePattern(ScenePrimitive),
    /// Shadow offset, spread, or color is invalid.
    InvalidShadow(ScenePrimitive),
    /// Active logical screen clip is non-finite or empty.
    InvalidScreenClip,
    /// Pseudo-depth must be finite.
    NonFiniteDepth,
    /// An accepted command would exceed an explicit scene work limit.
    BudgetExceeded {
        /// Work category whose limit was exceeded.
        resource: SceneBudgetResource,
        /// Configured maximum for the category.
        limit: usize,
        /// Total category usage that the command requested.
        requested: usize,
    },
    /// Engine-owned command storage could not be reserved without mutation.
    AllocationFailed {
        /// Minimum additional payload bytes associated with the insertion.
        requested_bytes: usize,
    },
}

impl fmt::Display for SceneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "scene command rejected: {self:?}")
    }
}

impl Error for SceneError {}

/// Ordered list of visual draw commands for a single frame.
///
/// A scene contains renderable primitives only. It does not own simulation
/// entities, domain rules, or time stepping.
#[derive(Debug, Clone)]
pub struct Scene {
    background: Color,
    /// Commands stored in draw order.
    commands: Vec<SceneCommand>,
    next_order: u64,
    current_screen_clip: Option<ScreenClipRect>,
    current_depth: f32,
    budget: Option<SceneBudget>,
    statistics: SceneStatistics,
}

impl Scene {
    /// Creates an empty scene with a normalized linear-RGBA background color.
    ///
    /// This compatibility constructor is intentionally unbounded. Production
    /// hosts handling externally sized visual state should prefer
    /// [`Scene::with_budget`].
    pub fn new(background: Color) -> Result<Self, SceneError> {
        Self::create(background, None)
    }

    /// Creates an empty scene with explicit construction and rendering limits.
    pub fn with_budget(background: Color, budget: SceneBudget) -> Result<Self, SceneError> {
        Self::create(background, Some(budget))
    }

    fn create(background: Color, budget: Option<SceneBudget>) -> Result<Self, SceneError> {
        if !background.is_normalized() {
            return Err(SceneError::InvalidBackground);
        }
        Ok(Self {
            background,
            commands: Vec::new(),
            next_order: 0,
            current_screen_clip: None,
            current_depth: 0.0,
            budget,
            statistics: SceneStatistics::default(),
        })
    }

    /// Returns the normalized clear color used before drawing commands.
    pub fn background(&self) -> Color {
        self.background
    }

    /// Replaces the clear color used before drawing commands.
    pub fn set_background(&mut self, background: Color) -> Result<(), SceneError> {
        if !background.is_normalized() {
            return Err(SceneError::InvalidBackground);
        }
        self.background = background;
        Ok(())
    }

    /// Removes all draw commands and active clipping without changing the background color.
    pub fn clear(&mut self) {
        self.commands.clear();
        self.next_order = 0;
        self.current_screen_clip = None;
        self.current_depth = 0.0;
        self.statistics = SceneStatistics::default();
    }

    /// Returns accepted commands in stable layer and insertion order.
    pub fn commands(&self) -> &[SceneCommand] {
        &self.commands
    }

    /// Returns the number of accepted render commands.
    pub fn command_count(&self) -> usize {
        self.commands.len()
    }

    /// Returns the explicit work budget, or `None` for an unbounded scene.
    pub const fn budget(&self) -> Option<SceneBudget> {
        self.budget
    }

    /// Returns requested, accepted, rejected, and estimated scene work.
    pub const fn statistics(&self) -> SceneStatistics {
        self.statistics
    }

    pub(crate) const fn current_screen_clip(&self) -> Option<ScreenClipRect> {
        self.current_screen_clip
    }

    /// Replaces the screen-space clipping rectangle captured by new commands.
    ///
    /// The rectangle uses logical screen pixels with the origin at the top-left
    /// of the render surface. Existing commands keep the clip they captured when
    /// they were appended. `None` disables clipping for subsequent commands.
    /// The previous clip is returned so callers can restore explicit drawing
    /// state without tracking it separately. Invalid clips are rejected here,
    /// before they can affect a later command insertion.
    pub fn set_screen_clip(
        &mut self,
        screen_clip: Option<ScreenClipRect>,
    ) -> Result<Option<ScreenClipRect>, SceneError> {
        if screen_clip.is_some_and(|clip| !clip.is_valid()) {
            return Err(SceneError::InvalidScreenClip);
        }
        Ok(std::mem::replace(
            &mut self.current_screen_clip,
            screen_clip,
        ))
    }

    /// Appends commands from `draw` using a temporary screen-space clip.
    ///
    /// The rectangle uses logical screen pixels. Nested clips are intersected,
    /// so inner drawing cannot escape an outer clip. The previous clip is
    /// restored after `draw` returns.
    pub fn with_screen_clip<R>(
        &mut self,
        screen_clip: ScreenClipRect,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, SceneError> {
        if !screen_clip.is_valid() {
            return Err(SceneError::InvalidScreenClip);
        }
        let previous = self.current_screen_clip;
        let combined = match previous {
            Some(outer) => outer.intersection(screen_clip)?,
            None => screen_clip,
        };
        if !combined.is_valid() {
            return Err(SceneError::InvalidScreenClip);
        }
        self.current_screen_clip = Some(combined);
        let result = catch_unwind(AssertUnwindSafe(|| draw(self)));
        self.current_screen_clip = previous;
        match result {
            Ok(result) => Ok(result),
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Replaces the pseudo-depth captured by subsequently appended commands.
    ///
    /// Depth is measured in caller-defined units and converted through
    /// [`crate::Projection2d::depth_scale`]. It affects projection only, not draw
    /// order. Non-finite values return `false` and leave the current depth unchanged.
    pub fn set_depth(&mut self, depth: f32) -> bool {
        self.try_set_depth(depth).is_ok()
    }

    /// Replaces pseudo-depth and reports why invalid input was rejected.
    pub fn try_set_depth(&mut self, depth: f32) -> Result<(), SceneError> {
        if !depth.is_finite() {
            return Err(SceneError::NonFiniteDepth);
        }
        self.current_depth = depth;
        Ok(())
    }

    /// Appends commands with a temporary pseudo-depth value.
    ///
    /// The previous depth is restored after `draw` returns or unwinds. Non-finite
    /// depth returns an error without calling `draw`.
    pub fn with_depth<R>(
        &mut self,
        depth: f32,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> Result<R, SceneError> {
        if !depth.is_finite() {
            return Err(SceneError::NonFiniteDepth);
        }

        let previous = self.current_depth;
        self.current_depth = depth;
        let result = catch_unwind(AssertUnwindSafe(|| draw(self)));
        self.current_depth = previous;
        match result {
            Ok(result) => Ok(result),
            Err(payload) => resume_unwind(payload),
        }
    }

    /// Appends a valid command to the default layer.
    ///
    /// Returns `false` and leaves the scene unchanged when the primitive contains
    /// non-finite coordinates, invalid dimensions, or no drawable geometry.
    pub fn push(&mut self, command: DrawCommand) -> bool {
        self.try_push(command).is_ok()
    }

    /// Appends a command to the default layer with structured rejection diagnostics.
    pub fn try_push(&mut self, command: DrawCommand) -> Result<(), SceneError> {
        self.try_push_to_layer(Layer::DEFAULT, command)
    }

    /// Appends a command to a layer.
    ///
    /// Lower layer values are drawn first. Commands on the same layer preserve
    /// insertion order.
    /// Returns `false` and leaves ordering unchanged when `command` is invalid.
    pub fn push_to_layer(&mut self, layer: Layer, command: DrawCommand) -> bool {
        self.try_push_to_layer(layer, command).is_ok()
    }

    /// Appends a command to a layer with structured rejection diagnostics.
    pub fn try_push_to_layer(
        &mut self,
        layer: Layer,
        command: DrawCommand,
    ) -> Result<(), SceneError> {
        let primitive = command.primitive();
        self.statistics.requested_commands = self.statistics.requested_commands.saturating_add(1);
        self.statistics.requested_by_primitive.increment(primitive);
        let result = self.try_push_to_layer_inner(layer, command);
        if result.is_err() {
            self.statistics.rejected_commands = self.statistics.rejected_commands.saturating_add(1);
            self.statistics.rejected_by_primitive.increment(primitive);
        }
        result
    }

    /// Atomically appends many commands and orders the combined scene once.
    ///
    /// This is the bounded high-volume construction path for adversarial or
    /// frequently alternating layers. Validation and budget accounting happen
    /// before the existing command list is changed. On failure, no command from
    /// the batch is retained; request and rejection diagnostics still advance.
    /// The resulting order is stable by layer and by batch insertion order.
    pub fn try_extend_to_layers(
        &mut self,
        commands: impl IntoIterator<Item = (Layer, DrawCommand)>,
    ) -> Result<(), SceneError> {
        let mut staged = Vec::new();
        let mut requested = self.statistics;
        let mut attempted = 0usize;
        let mut attempted_by_primitive = PrimitiveCommandCounts::default();
        let mut next_order = self.next_order;

        for (layer, command) in commands {
            let primitive = command.primitive();
            attempted = attempted.saturating_add(1);
            attempted_by_primitive.increment(primitive);
            requested.requested_commands = requested.requested_commands.saturating_add(1);
            requested.requested_by_primitive.increment(primitive);
            if let Err(error) = command.validate() {
                self.record_batch_rejection(attempted, attempted_by_primitive);
                return Err(error);
            }
            if self
                .current_screen_clip
                .is_some_and(|screen_clip| !screen_clip.is_valid())
            {
                self.record_batch_rejection(attempted, attempted_by_primitive);
                return Err(SceneError::InvalidScreenClip);
            }

            requested = requested.with_command(&command);
            if let Some(budget) = self.budget
                && let Err(error) = validate_scene_budget(budget, requested)
            {
                self.record_batch_rejection(attempted, attempted_by_primitive);
                return Err(error);
            }
            if staged.try_reserve(1).is_err() {
                self.record_batch_rejection(attempted, attempted_by_primitive);
                return Err(SceneError::AllocationFailed {
                    requested_bytes: command.retained_bytes(),
                });
            }
            staged.push(SceneCommand {
                layer,
                order: next_order,
                depth: self.current_depth,
                screen_clip: self.current_screen_clip,
                command,
            });
            next_order = next_order.saturating_add(1);
        }

        if staged.is_empty() {
            return Ok(());
        }
        let total = self.commands.len().saturating_add(staged.len());
        let mut merged = Vec::new();
        if merged.try_reserve(total).is_err() {
            self.record_batch_rejection(attempted, attempted_by_primitive);
            return Err(SceneError::AllocationFailed {
                requested_bytes: total.saturating_mul(size_of::<SceneCommand>()),
            });
        }
        merged.append(&mut self.commands);
        merged.append(&mut staged);
        merged.sort_unstable_by_key(|command| (command.layer, command.order));
        self.commands = merged;
        self.next_order = next_order;
        self.statistics = requested;
        Ok(())
    }

    fn record_batch_rejection(
        &mut self,
        command_count: usize,
        by_primitive: PrimitiveCommandCounts,
    ) {
        self.statistics.requested_commands = self
            .statistics
            .requested_commands
            .saturating_add(command_count);
        self.statistics.rejected_commands = self
            .statistics
            .rejected_commands
            .saturating_add(command_count);
        self.statistics.requested_by_primitive =
            self.statistics.requested_by_primitive.adding(by_primitive);
        self.statistics.rejected_by_primitive =
            self.statistics.rejected_by_primitive.adding(by_primitive);
    }

    pub(crate) fn record_external_rejection(&mut self, primitive: ScenePrimitive) {
        let mut counts = PrimitiveCommandCounts::default();
        counts.increment(primitive);
        self.record_batch_rejection(1, counts);
    }

    fn try_push_to_layer_inner(
        &mut self,
        layer: Layer,
        command: DrawCommand,
    ) -> Result<(), SceneError> {
        command.validate()?;
        if self
            .current_screen_clip
            .is_some_and(|screen_clip| !screen_clip.is_valid())
        {
            return Err(SceneError::InvalidScreenClip);
        }

        let command_retained_bytes = command.retained_bytes();
        let requested = self.statistics.with_command(&command);
        if let Some(budget) = self.budget {
            validate_scene_budget(budget, requested)?;
        }

        self.commands
            .try_reserve(1)
            .map_err(|_| SceneError::AllocationFailed {
                requested_bytes: command_retained_bytes,
            })?;

        let order = self.next_order;
        let scene_command = SceneCommand {
            layer,
            order,
            depth: self.current_depth,
            screen_clip: self.current_screen_clip,
            command,
        };
        let position = self.commands.partition_point(|existing| {
            existing.layer < scene_command.layer
                || (existing.layer == scene_command.layer && existing.order <= scene_command.order)
        });
        self.commands.insert(position, scene_command);
        self.next_order = self.next_order.saturating_add(1);
        self.statistics = requested;
        Ok(())
    }

    /// Appends a circle in world coordinates.
    ///
    /// `radius` is in world units. Returns `false` for non-positive or non-finite
    /// radius, non-finite center, or invalid style values.
    pub fn circle(&mut self, center: Vec2, radius: f32, style: ShapeStyle) -> bool {
        self.try_circle(center, radius, style).is_ok()
    }

    /// Appends a circle and returns a structured validation error on rejection.
    pub fn try_circle(
        &mut self,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_circle_on_layer(Layer::DEFAULT, center, radius, style)
    }

    /// Appends a circle in world coordinates to a layer.
    ///
    /// `radius` is in world units. Returns `false` for non-positive or non-finite
    /// radius, non-finite center, or invalid style values.
    pub fn circle_on_layer(
        &mut self,
        layer: Layer,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> bool {
        self.try_circle_on_layer(layer, center, radius, style)
            .is_ok()
    }

    /// Appends a circle to a layer with structured rejection diagnostics.
    pub fn try_circle_on_layer(
        &mut self,
        layer: Layer,
        center: Vec2,
        radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Circle(Circle {
                center,
                radius,
                style,
            }),
        )
    }

    /// Appends an axis-aligned rectangle in world coordinates.
    ///
    /// `corner_radius` is in world units and renderers clamp it to half the
    /// rectangle size. Invalid bounds, negative radius, and invalid styles return
    /// `false` without adding a command.
    pub fn rect(&mut self, rect: Rect, corner_radius: f32, style: ShapeStyle) -> bool {
        self.try_rect(rect, corner_radius, style).is_ok()
    }

    /// Appends a rectangle and returns a structured validation error on rejection.
    pub fn try_rect(
        &mut self,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_rect_on_layer(Layer::DEFAULT, rect, corner_radius, style)
    }

    /// Appends an axis-aligned rectangle in world coordinates to a layer.
    ///
    /// `corner_radius` is in world units and renderers clamp it to half the
    /// rectangle size. Invalid bounds, negative radius, and invalid styles return
    /// `false` without adding a command.
    pub fn rect_on_layer(
        &mut self,
        layer: Layer,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> bool {
        self.try_rect_on_layer(layer, rect, corner_radius, style)
            .is_ok()
    }

    /// Appends a rectangle to a layer with structured rejection diagnostics.
    pub fn try_rect_on_layer(
        &mut self,
        layer: Layer,
        rect: Rect,
        corner_radius: f32,
        style: ShapeStyle,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(
            layer,
            DrawCommand::Rect(RectShape {
                rect,
                corner_radius,
                style,
            }),
        )
    }

    /// Appends a stroked line between world-space points.
    ///
    /// `width` is in logical screen pixels so line thickness stays readable while the
    /// camera zooms. Degenerate geometry and invalid width or color return
    /// `false`.
    pub fn line(&mut self, from: Vec2, to: Vec2, width: f32, color: Color) -> bool {
        self.try_line(from, to, width, color).is_ok()
    }

    /// Appends a line and returns a structured validation error on rejection.
    pub fn try_line(
        &mut self,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_styled_line(from, to, StrokeStyle2d::new(width, color))
    }

    /// Appends a stroked line between world-space points to a layer.
    ///
    /// `width` is in logical screen pixels so line thickness stays readable while the
    /// camera zooms. Degenerate geometry and invalid width or color return
    /// `false`.
    pub fn line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> bool {
        self.try_line_on_layer(layer, from, to, width, color)
            .is_ok()
    }

    /// Appends a line to a layer with structured rejection diagnostics.
    pub fn try_line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_styled_line_on_layer(layer, from, to, StrokeStyle2d::new(width, color))
    }

    /// Appends a line with explicit width units, caps, joins, dashes, and markers.
    pub fn styled_line(&mut self, from: Vec2, to: Vec2, style: StrokeStyle2d) -> bool {
        self.try_styled_line(from, to, style).is_ok()
    }

    /// Appends an explicitly styled line with structured rejection diagnostics.
    pub fn try_styled_line(
        &mut self,
        from: Vec2,
        to: Vec2,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_styled_line_on_layer(Layer::DEFAULT, from, to, style)
    }

    /// Appends an explicitly styled line to a shared draw layer.
    pub fn styled_line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        style: StrokeStyle2d,
    ) -> bool {
        self.try_styled_line_on_layer(layer, from, to, style)
            .is_ok()
    }

    /// Appends an explicitly styled line to a layer with structured rejection.
    pub fn try_styled_line_on_layer(
        &mut self,
        layer: Layer,
        from: Vec2,
        to: Vec2,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(layer, DrawCommand::Line(Line { from, to, style }))
    }

    /// Appends connected stroked segments through world-space points.
    ///
    /// `width` is in logical screen pixels. Empty, single-point, fully degenerate, and
    /// non-finite input returns `false` without adding a command.
    pub fn polyline(&mut self, points: Vec<Vec2>, width: f32, color: Color) -> bool {
        self.try_polyline(points, width, color).is_ok()
    }

    /// Appends a polyline and returns a structured validation error on rejection.
    pub fn try_polyline(
        &mut self,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_styled_polyline(points, StrokeStyle2d::new(width, color))
    }

    /// Appends connected stroked segments through world-space points to a layer.
    ///
    /// `width` is in logical screen pixels. Empty, single-point, fully degenerate, and
    /// non-finite input returns `false` without adding a command.
    pub fn polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> bool {
        self.try_polyline_on_layer(layer, points, width, color)
            .is_ok()
    }

    /// Appends a polyline to a layer with structured rejection diagnostics.
    pub fn try_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        width: f32,
        color: Color,
    ) -> Result<(), SceneError> {
        self.try_styled_polyline_on_layer(layer, points, StrokeStyle2d::new(width, color))
    }

    /// Appends a connected path with explicit width, caps, joins, dashes, and markers.
    pub fn styled_polyline(&mut self, points: Vec<Vec2>, style: StrokeStyle2d) -> bool {
        self.try_styled_polyline(points, style).is_ok()
    }

    /// Appends an explicitly styled path with structured rejection diagnostics.
    pub fn try_styled_polyline(
        &mut self,
        points: Vec<Vec2>,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_styled_polyline_on_layer(Layer::DEFAULT, points, style)
    }

    /// Appends an explicitly styled path to a shared draw layer.
    pub fn styled_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        style: StrokeStyle2d,
    ) -> bool {
        self.try_styled_polyline_on_layer(layer, points, style)
            .is_ok()
    }

    /// Appends an explicitly styled path to a layer with structured rejection.
    pub fn try_styled_polyline_on_layer(
        &mut self,
        layer: Layer,
        points: Vec<Vec2>,
        style: StrokeStyle2d,
    ) -> Result<(), SceneError> {
        self.try_push_to_layer(layer, DrawCommand::Polyline(Polyline { points, style }))
    }
}

/// Draw layer used to order scene commands.
///
/// Lower values are drawn first. Higher values appear above lower layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layer {
    order: i32,
}

impl Layer {
    /// Default layer used by convenience drawing methods.
    pub const DEFAULT: Self = Self::new(0);
    /// Conventional layer for background guides or grids.
    pub const BACKGROUND: Self = Self::new(-100);
    /// Conventional layer for foreground annotations or overlays.
    pub const FOREGROUND: Self = Self::new(100);

    /// Builds a layer from an order value.
    pub const fn new(order: i32) -> Self {
        Self { order }
    }

    /// Returns the order value used for sorting.
    pub const fn order(self) -> i32 {
        self.order
    }
}

/// Axis-aligned clipping rectangle in logical screen pixels.
///
/// The coordinate origin is the top-left of the render surface and positive y
/// points downward. Constructors reject empty and non-finite bounds.
/// Renderers clamp valid clips to the target viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenClipRect {
    rect: Rect,
}

impl ScreenClipRect {
    /// Builds a non-empty clip from two corners in logical screen pixels.
    pub fn new(min: LogicalScreenPosition, max: LogicalScreenPosition) -> Result<Self, SceneError> {
        Self::from_rect(Rect::new(min.to_vec2(), max.to_vec2()))
    }

    /// Builds a clip from its top-left corner and size in logical screen pixels.
    ///
    /// Negative size components are accepted and normalized before validation.
    pub fn from_min_size(
        min: LogicalScreenPosition,
        size: LogicalScreenVector,
    ) -> Result<Self, SceneError> {
        Self::from_rect(Rect::from_min_size(min.to_vec2(), size.to_vec2()))
    }

    fn from_rect(rect: Rect) -> Result<Self, SceneError> {
        let clip = Self { rect };
        clip.is_valid()
            .then_some(clip)
            .ok_or(SceneError::InvalidScreenClip)
    }

    #[cfg(feature = "wgpu")]
    pub(crate) fn rect(self) -> Rect {
        self.rect
    }

    /// Returns the normalized top-left corner in logical screen pixels.
    pub fn min(self) -> LogicalScreenPosition {
        LogicalScreenPosition::from_vec2(self.rect.normalized().min)
    }

    /// Returns the normalized bottom-right corner in logical screen pixels.
    pub fn max(self) -> LogicalScreenPosition {
        LogicalScreenPosition::from_vec2(self.rect.normalized().max)
    }

    /// Returns the overlap between two screen-space clips.
    ///
    /// Disjoint inputs return [`SceneError::InvalidScreenClip`].
    pub fn intersection(self, other: Self) -> Result<Self, SceneError> {
        let first = self.rect.normalized();
        let second = other.rect.normalized();
        let min = Vec2::new(first.min.x.max(second.min.x), first.min.y.max(second.min.y));
        let max = Vec2::new(first.max.x.min(second.max.x), first.max.y.min(second.max.y));

        Self::new(
            LogicalScreenPosition::from_vec2(min),
            LogicalScreenPosition::from_vec2(Vec2::new(max.x.max(min.x), max.y.max(min.y))),
        )
    }

    fn is_valid(self) -> bool {
        if !self.rect.min.is_finite() || !self.rect.max.is_finite() {
            return false;
        }

        let rect = self.rect.normalized();
        rect.width() > 0.0 && rect.height() > 0.0
    }
}

/// Draw command with layer and stable insertion order metadata.
#[derive(Debug, Clone)]
pub struct SceneCommand {
    /// Layer used to sort commands before rendering.
    layer: Layer,
    /// Monotonic insertion order within the scene.
    order: u64,
    /// Caller-defined pseudo-depth used by camera projection.
    depth: f32,
    /// Optional clipping rectangle in logical screen pixels.
    screen_clip: Option<ScreenClipRect>,
    /// Visual primitive to render.
    command: DrawCommand,
}

impl SceneCommand {
    /// Returns the layer used for stable scene ordering.
    pub fn layer(&self) -> Layer {
        self.layer
    }

    /// Returns the insertion order assigned by the scene.
    pub fn order(&self) -> u64 {
        self.order
    }

    /// Returns caller-defined pseudo-depth captured by this command.
    ///
    /// Depth affects camera projection only. Commands on one layer retain their
    /// insertion order regardless of depth.
    pub fn depth(&self) -> f32 {
        self.depth
    }

    /// Returns the optional clip captured when the command was appended.
    pub fn screen_clip(&self) -> Option<ScreenClipRect> {
        self.screen_clip
    }

    /// Returns the visual primitive stored by this command.
    pub fn command(&self) -> &DrawCommand {
        &self.command
    }
}

/// Renderable primitive command.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DrawCommand {
    /// Filled and/or stroked circle.
    Circle(Circle),
    /// Filled and/or stroked rectangle.
    Rect(RectShape),
    /// Single stroked segment.
    Line(Line),
    /// Connected stroked segments.
    Polyline(Polyline),
}

impl DrawCommand {
    /// Returns the primitive category used by diagnostics and budgets.
    pub const fn primitive(&self) -> ScenePrimitive {
        match self {
            Self::Circle(_) => ScenePrimitive::Circle,
            Self::Rect(_) => ScenePrimitive::Rect,
            Self::Line(_) => ScenePrimitive::Line,
            Self::Polyline(_) => ScenePrimitive::Polyline,
        }
    }

    /// Builds and validates a circle command for batch insertion.
    pub fn circle(center: Vec2, radius: f32, style: ShapeStyle) -> Result<Self, SceneError> {
        let command = Self::Circle(Circle {
            center,
            radius,
            style,
        });
        command.validate()?;
        Ok(command)
    }

    /// Builds and validates a rectangle command for batch insertion.
    pub fn rect(rect: Rect, corner_radius: f32, style: ShapeStyle) -> Result<Self, SceneError> {
        let command = Self::Rect(RectShape {
            rect,
            corner_radius,
            style,
        });
        command.validate()?;
        Ok(command)
    }

    /// Builds and validates a single line command for batch insertion.
    pub fn line(from: Vec2, to: Vec2, width: f32, color: Color) -> Result<Self, SceneError> {
        Self::styled_line(from, to, StrokeStyle2d::new(width, color))
    }

    /// Builds and validates an explicitly styled line command.
    pub fn styled_line(from: Vec2, to: Vec2, style: StrokeStyle2d) -> Result<Self, SceneError> {
        let command = Self::Line(Line { from, to, style });
        command.validate()?;
        Ok(command)
    }

    /// Builds and validates a connected polyline command for batch insertion.
    pub fn polyline(points: Vec<Vec2>, width: f32, color: Color) -> Result<Self, SceneError> {
        Self::styled_polyline(points, StrokeStyle2d::new(width, color))
    }

    /// Builds and validates an explicitly styled polyline command.
    pub fn styled_polyline(points: Vec<Vec2>, style: StrokeStyle2d) -> Result<Self, SceneError> {
        let command = Self::Polyline(Polyline { points, style });
        command.validate()?;
        Ok(command)
    }

    fn validate(&self) -> Result<(), SceneError> {
        match self {
            Self::Circle(circle) => {
                if !circle.center.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle));
                }
                if !circle.radius.is_finite() || circle.radius <= 0.0 {
                    return Err(SceneError::InvalidDimension(ScenePrimitive::Circle));
                }
                if !(circle.center + Vec2::splat(circle.radius)).is_finite()
                    || !(circle.center - Vec2::splat(circle.radius)).is_finite()
                {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle));
                }
                circle.style.validate(ScenePrimitive::Circle)
            }
            Self::Rect(rectangle) => {
                let rect = rectangle.rect.normalized();
                if !rect.min.is_finite() || !rect.max.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
                }
                let width = rect.width();
                let height = rect.height();
                if !width.is_finite() || !height.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect));
                }
                if width <= 0.0
                    || height <= 0.0
                    || !rectangle.corner_radius.is_finite()
                    || rectangle.corner_radius < 0.0
                {
                    return Err(SceneError::InvalidDimension(ScenePrimitive::Rect));
                }
                rectangle.style.validate(ScenePrimitive::Rect)
            }
            Self::Line(line) => {
                if !line.from.is_finite() || !line.to.is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line));
                }
                if !(line.to - line.from).is_finite() {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line));
                }
                if !drawable_segment(line.from, line.to) {
                    return Err(SceneError::DegenerateGeometry(ScenePrimitive::Line));
                }
                if !line.style.is_valid() {
                    return Err(SceneError::InvalidStroke(ScenePrimitive::Line));
                }
                validate_stroke_path(&[line.from, line.to], line.style, ScenePrimitive::Line)?;
                Ok(())
            }
            Self::Polyline(polyline) => {
                if !polyline.points.iter().all(|point| point.is_finite()) {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Polyline));
                }
                if polyline
                    .points
                    .windows(2)
                    .any(|pair| !(pair[1] - pair[0]).is_finite())
                {
                    return Err(SceneError::NonFiniteGeometry(ScenePrimitive::Polyline));
                }
                if polyline.points.len() < 2
                    || !polyline
                        .points
                        .windows(2)
                        .any(|pair| drawable_segment(pair[0], pair[1]))
                {
                    return Err(SceneError::DegenerateGeometry(ScenePrimitive::Polyline));
                }
                if !polyline.style.is_valid() {
                    return Err(SceneError::InvalidStroke(ScenePrimitive::Polyline));
                }
                validate_stroke_path(&polyline.points, polyline.style, ScenePrimitive::Polyline)?;
                Ok(())
            }
        }
    }

    fn retained_point_count(&self) -> usize {
        match self {
            Self::Polyline(polyline) => polyline.points.len(),
            Self::Circle(_) | Self::Rect(_) | Self::Line(_) => 0,
        }
    }

    fn retained_bytes(&self) -> usize {
        let point_bytes = match self {
            Self::Polyline(polyline) => {
                polyline.points.capacity().saturating_mul(size_of::<Vec2>())
            }
            Self::Circle(_) | Self::Rect(_) | Self::Line(_) => 0,
        };
        size_of::<SceneCommand>().saturating_add(point_bytes)
    }

    fn estimated_tessellated_vertices(&self) -> usize {
        match self {
            Self::Circle(circle) => {
                let fill_vertices = CIRCLE_SEGMENTS * 3;
                let stroke_vertices = CIRCLE_SEGMENTS * 6;
                let style = circle.style;
                usize::from(style.fill.is_some()) * fill_vertices
                    + usize::from(style.stroke.is_some()) * stroke_vertices
                    + style.shadow.map_or(0, |shadow| {
                        fill_vertices + usize::from(shadow.spread > 0.0) * stroke_vertices
                    })
            }
            Self::Rect(rectangle) => {
                let rect = rectangle.rect.normalized();
                let radius = rectangle
                    .corner_radius
                    .max(0.0)
                    .min(rect.width().abs() * 0.5)
                    .min(rect.height().abs() * 0.5);
                let boundary_segments = if radius <= 0.0 {
                    4
                } else {
                    4 * (CORNER_SEGMENTS + 1)
                };
                let fill_vertices = boundary_segments * 3;
                let stroke_vertices = boundary_segments * 6;
                let style = rectangle.style;
                usize::from(style.fill.is_some()) * fill_vertices
                    + usize::from(style.stroke.is_some()) * stroke_vertices
                    + style.shadow.map_or(0, |shadow| {
                        fill_vertices + usize::from(shadow.spread > 0.0) * stroke_vertices
                    })
            }
            Self::Line(line) => estimate_stroke_vertices(&[line.from, line.to], line.style),
            Self::Polyline(polyline) => estimate_stroke_vertices(&polyline.points, polyline.style),
        }
    }
}

fn validate_stroke_path(
    points: &[Vec2],
    style: StrokeStyle2d,
    primitive: ScenePrimitive,
) -> Result<(), SceneError> {
    if points
        .windows(2)
        .any(|pair| !drawable_segment(pair[0], pair[1]))
    {
        return Err(SceneError::DegenerateGeometry(primitive));
    }
    for (index, window) in points.windows(3).enumerate() {
        let incoming = window[1] - window[0];
        let outgoing = window[2] - window[1];
        let cross = f64::from(incoming.x) * f64::from(outgoing.y)
            - f64::from(incoming.y) * f64::from(outgoing.x);
        let dot = f64::from(incoming.x) * f64::from(outgoing.x)
            + f64::from(incoming.y) * f64::from(outgoing.y);
        let direction_product = f64::from(incoming.x).hypot(f64::from(incoming.y))
            * f64::from(outgoing.x).hypot(f64::from(outgoing.y));
        let normalized_cross = cross / direction_product;
        if normalized_cross.abs() <= MIN_STROKE_TURN_SINE && dot < 0.0 {
            return Err(SceneError::DegenerateStrokeTurn {
                primitive,
                vertex_index: index + 1,
            });
        }
    }
    let maximum_offset = f64::from(style.stroke.width) * 0.5 * f64::from(style.miter_limit);
    if !maximum_offset.is_finite() || maximum_offset > f64::from(f32::MAX) {
        return Err(SceneError::InvalidStroke(primitive));
    }
    if style.width_mode == StrokeWidthMode2d::WorldUnits {
        let extent = Vec2::splat(maximum_offset as f32);
        if points
            .iter()
            .any(|point| !(*point + extent).is_finite() || !(*point - extent).is_finite())
        {
            return Err(SceneError::NonFiniteGeometry(primitive));
        }
    }
    if let Some(dash) = style.dash {
        let required = dash_subsegment_count(points, dash, Some(dash.max_subsegments))
            .ok_or(SceneError::UnrepresentableStrokePattern(primitive))?;
        if required > dash.max_subsegments {
            return Err(SceneError::StrokeExpansionLimitExceeded {
                primitive,
                limit: dash.max_subsegments,
                required,
            });
        }
    }
    Ok(())
}

fn estimate_stroke_vertices(points: &[Vec2], style: StrokeStyle2d) -> usize {
    let segment_count = points
        .windows(2)
        .filter(|pair| drawable_segment(pair[0], pair[1]))
        .count();
    if segment_count == 0 {
        return 0;
    }
    let body_count = style.dash.map_or(segment_count, |dash| {
        dash_subsegment_count(points, dash, None)
            .expect("accepted dash paths have representable subsegments")
    });
    // A visible dash may continue across every geometric vertex. Counting all
    // possible joins is conservative without expanding the pattern twice.
    let join_count = segment_count.saturating_sub(1);
    let cap_count = if style.dash.is_some() {
        body_count.saturating_mul(2)
    } else {
        2
    };
    let cap_vertices = match style.cap {
        StrokeCap2d::Round => cap_count.saturating_mul(ROUND_CAP_SEGMENTS * 3),
        StrokeCap2d::Butt | StrokeCap2d::Square => 0,
    };
    let join_vertices = match style.join {
        StrokeJoin2d::Bevel => join_count.saturating_mul(6),
        StrokeJoin2d::Miter => join_count.saturating_mul(12),
        // Logical round joins carry one candidate fan for each possible screen
        // orientation; the shader collapses the inactive fan to zero area.
        StrokeJoin2d::Round if style.width_mode == StrokeWidthMode2d::LogicalPixels => {
            join_count.saturating_mul(ROUND_CAP_SEGMENTS * 6)
        }
        StrokeJoin2d::Round => join_count.saturating_mul(ROUND_CAP_SEGMENTS * 3),
    };
    let marker_vertices = (usize::from(style.start_marker.is_some())
        + usize::from(style.end_marker.is_some()))
    .saturating_mul(3);
    body_count
        .saturating_mul(6)
        .saturating_add(cap_vertices)
        .saturating_add(join_vertices)
        .saturating_add(marker_vertices)
}

fn dash_subsegment_count(
    points: &[Vec2],
    dash: StrokeDashPattern2d,
    stop_after: Option<usize>,
) -> Option<usize> {
    let lengths = dash.lengths();
    let total: f64 = lengths.iter().map(|length| f64::from(*length)).sum();
    let mut phase = f64::from(dash.phase).rem_euclid(total);
    let mut pattern_index = 0usize;
    while phase >= f64::from(lengths[pattern_index]) {
        phase -= f64::from(lengths[pattern_index]);
        pattern_index = (pattern_index + 1) % lengths.len();
    }
    let mut pattern_remaining = f64::from(lengths[pattern_index]) - phase;
    let mut count = 0usize;

    for pair in points.windows(2) {
        let delta = pair[1] - pair[0];
        let segment_length = f64::from(delta.x).hypot(f64::from(delta.y));
        let mut segment_remaining = segment_length;
        while segment_remaining > 0.0 {
            let consumed = segment_remaining.min(pattern_remaining);
            if pattern_index.is_multiple_of(2) && consumed > 0.0 {
                count = count.saturating_add(1);
                if stop_after.is_some_and(|limit| count > limit) {
                    return Some(count);
                }
            }
            let next_remaining = segment_remaining - consumed;
            if next_remaining >= segment_remaining {
                return None;
            }
            let amount_start = 1.0 - segment_remaining / segment_length;
            let amount_end = 1.0 - next_remaining / segment_length;
            if dash_segment_point(pair[0], pair[1], amount_start)
                == dash_segment_point(pair[0], pair[1], amount_end)
            {
                return None;
            }
            segment_remaining = next_remaining;
            pattern_remaining -= consumed;
            if pattern_remaining <= 0.0 {
                pattern_index = (pattern_index + 1) % lengths.len();
                pattern_remaining = f64::from(lengths[pattern_index]);
            }
        }
    }
    Some(count)
}

fn dash_segment_point(from: Vec2, to: Vec2, amount: f64) -> Vec2 {
    Vec2::new(
        (f64::from(from.x) * (1.0 - amount) + f64::from(to.x) * amount) as f32,
        (f64::from(from.y) * (1.0 - amount) + f64::from(to.y) * amount) as f32,
    )
}

fn validate_scene_budget(
    budget: SceneBudget,
    requested: SceneStatistics,
) -> Result<(), SceneError> {
    let limits = [
        (
            SceneBudgetResource::Commands,
            budget.max_commands,
            requested.accepted_commands,
        ),
        (
            SceneBudgetResource::Points,
            budget.max_points,
            requested.retained_points,
        ),
        (
            SceneBudgetResource::TessellatedVertices,
            budget.max_tessellated_vertices,
            requested.estimated_tessellated_vertices,
        ),
        (
            SceneBudgetResource::RetainedBytes,
            budget.max_retained_bytes,
            requested.retained_bytes,
        ),
        (
            SceneBudgetResource::UploadBytes,
            budget.max_upload_bytes,
            requested.estimated_upload_bytes,
        ),
        (
            SceneBudgetResource::DrawBatches,
            budget.max_draw_batches,
            requested.estimated_draw_batches,
        ),
    ];
    for (resource, limit, requested) in limits {
        if requested > limit {
            return Err(SceneError::BudgetExceeded {
                resource,
                limit,
                requested,
            });
        }
    }
    Ok(())
}

fn drawable_segment(from: Vec2, to: Vec2) -> bool {
    let delta = to - from;
    delta.is_finite() && (delta.x != 0.0 || delta.y != 0.0)
}

/// Circle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Circle {
    center: Vec2,
    radius: f32,
    style: ShapeStyle,
}

impl Circle {
    /// Returns the center in world coordinates.
    pub const fn center(&self) -> Vec2 {
        self.center
    }
    /// Returns the radius in world units.
    pub const fn radius(&self) -> f32 {
        self.radius
    }
    /// Returns fill, stroke, and shadow styling.
    pub const fn style(&self) -> ShapeStyle {
        self.style
    }
}

/// Axis-aligned rectangle primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct RectShape {
    rect: Rect,
    corner_radius: f32,
    style: ShapeStyle,
}

impl RectShape {
    /// Returns rectangle bounds in world coordinates.
    pub const fn rect(&self) -> Rect {
        self.rect
    }
    /// Returns corner radius in world units.
    pub const fn corner_radius(&self) -> f32 {
        self.corner_radius
    }
    /// Returns fill, stroke, and shadow styling.
    pub const fn style(&self) -> ShapeStyle {
        self.style
    }
}

/// Stroked line primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Line {
    from: Vec2,
    to: Vec2,
    style: StrokeStyle2d,
}

impl Line {
    /// Returns the start point in world coordinates.
    pub const fn from(&self) -> Vec2 {
        self.from
    }
    /// Returns the end point in world coordinates.
    pub const fn to(&self) -> Vec2 {
        self.to
    }
    /// Returns color and scalar width; inspect [`Self::stroke_style`] for units.
    pub const fn stroke(&self) -> Stroke {
        self.style.stroke
    }
    /// Returns the complete cap, join, dash, marker, and width-mode style.
    pub const fn stroke_style(&self) -> StrokeStyle2d {
        self.style
    }
}

/// Connected line-strip primitive in world coordinates.
#[derive(Debug, Clone)]
pub struct Polyline {
    points: Vec<Vec2>,
    style: StrokeStyle2d,
}

impl Polyline {
    /// Returns connected points in world coordinates.
    pub fn points(&self) -> &[Vec2] {
        &self.points
    }
    /// Returns color and scalar width; inspect [`Self::stroke_style`] for units.
    pub const fn stroke(&self) -> Stroke {
        self.style.stroke
    }
    /// Returns the complete cap, join, dash, marker, and width-mode style.
    pub const fn stroke_style(&self) -> StrokeStyle2d {
        self.style
    }
}

/// Coordinate space used by a line or polyline stroke width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeWidthMode2d {
    /// Width remains constant in logical screen pixels while the camera zooms.
    LogicalPixels,
    /// Width is measured in the same world units as the path coordinates.
    WorldUnits,
}

/// Presentation used at an open line or dash endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeCap2d {
    /// Stop exactly at the endpoint.
    Butt,
    /// Extend by half the stroke width beyond the endpoint.
    Square,
    /// Add a semicircular endpoint.
    Round,
}

/// Presentation used where two visible polyline segments meet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeJoin2d {
    /// Connect the two segment corners directly.
    Bevel,
    /// Fill the corner with a circular join.
    Round,
    /// Extend corner edges to their intersection, bounded by the style's
    /// miter limit and falling back to a bevel beyond it.
    Miter,
}

/// Fixed-size, allocation-free dash pattern in source path-coordinate units.
///
/// Values alternate visible and hidden lengths, beginning with a visible
/// length. The element count must be even and at most eight. `phase` advances
/// into the repeating pattern in the same path units. `max_subsegments` is a
/// hard limit on visible pieces generated for one command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeDashPattern2d {
    lengths: [f32; MAX_DASH_ELEMENTS],
    length_count: u8,
    phase: f32,
    max_subsegments: usize,
}

impl StrokeDashPattern2d {
    /// Builds a bounded alternating visible/hidden dash pattern.
    pub fn new(
        lengths: &[f32],
        phase: f32,
        max_subsegments: usize,
    ) -> Result<Self, StrokeStyleError> {
        if lengths.len() < 2
            || lengths.len() > MAX_DASH_ELEMENTS
            || !lengths.len().is_multiple_of(2)
        {
            return Err(StrokeStyleError::InvalidDashElementCount);
        }
        if lengths
            .iter()
            .any(|length| !length.is_finite() || *length <= 0.0)
        {
            return Err(StrokeStyleError::InvalidDashLength);
        }
        if !phase.is_finite() {
            return Err(StrokeStyleError::InvalidDashPhase);
        }
        if max_subsegments == 0 || max_subsegments > MAX_STROKE_DASH_SUBSEGMENTS {
            return Err(StrokeStyleError::InvalidDashExpansionLimit);
        }
        let mut stored = [0.0; MAX_DASH_ELEMENTS];
        stored[..lengths.len()].copy_from_slice(lengths);
        Ok(Self {
            lengths: stored,
            length_count: lengths.len() as u8,
            phase,
            max_subsegments,
        })
    }

    /// Returns the alternating visible/hidden lengths in path units.
    pub fn lengths(&self) -> &[f32] {
        &self.lengths[..usize::from(self.length_count)]
    }

    /// Returns the repeating-pattern phase in path units.
    pub const fn phase(self) -> f32 {
        self.phase
    }

    /// Returns the hard visible-subsegment expansion limit.
    pub const fn max_subsegments(self) -> usize {
        self.max_subsegments
    }
}

/// Reusable arrow marker definition for a line endpoint.
///
/// Marker dimensions are logical pixels even when the line body uses a world
/// width, keeping scientific annotations readable under camera zoom. A marked
/// endpoint ignores the ordinary cap: the body ends with a butt boundary at
/// the path endpoint, which is also the marker base. The filled triangle grows
/// outward from the path, so markers remain interior-disjoint from arbitrarily
/// short, dashed, or camera-scaled terminal segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeMarker2d {
    length: LogicalPixels,
    width: LogicalPixels,
}

impl StrokeMarker2d {
    /// Builds a filled triangular arrow marker.
    pub const fn arrow(length: LogicalPixels, width: LogicalPixels) -> Self {
        Self { length, width }
    }

    /// Returns marker length along the endpoint tangent.
    pub const fn length(self) -> LogicalPixels {
        self.length
    }

    /// Returns full marker width perpendicular to the endpoint tangent.
    pub const fn width(self) -> LogicalPixels {
        self.width
    }
}

/// Complete bounded styling for open 2D lines and polylines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrokeStyle2d {
    stroke: Stroke,
    width_mode: StrokeWidthMode2d,
    cap: StrokeCap2d,
    join: StrokeJoin2d,
    miter_limit: f32,
    dash: Option<StrokeDashPattern2d>,
    start_marker: Option<StrokeMarker2d>,
    end_marker: Option<StrokeMarker2d>,
}

impl StrokeStyle2d {
    /// Builds the legacy logical-pixel round-cap joined-strip style.
    pub const fn new(width: f32, color: Color) -> Self {
        Self {
            stroke: Stroke::new(width, color),
            width_mode: StrokeWidthMode2d::LogicalPixels,
            cap: StrokeCap2d::Round,
            join: StrokeJoin2d::Miter,
            miter_limit: MAX_MITER_LIMIT,
            dash: None,
            start_marker: None,
            end_marker: None,
        }
    }

    /// Builds a validated logical-pixel-width style.
    pub const fn logical(width: LogicalPixels, color: Color) -> Self {
        Self::new(width.get(), color)
    }

    /// Builds a validated world-unit-width style.
    pub const fn world(width: WorldLength, color: Color) -> Self {
        let mut style = Self::new(width.get(), color);
        style.width_mode = StrokeWidthMode2d::WorldUnits;
        style
    }

    /// Selects the open endpoint presentation.
    pub const fn with_cap(mut self, cap: StrokeCap2d) -> Self {
        self.cap = cap;
        self
    }

    /// Selects the connected-segment presentation.
    pub const fn with_join(mut self, join: StrokeJoin2d) -> Self {
        self.join = join;
        self
    }

    /// Sets the miter-length multiple in `1.0..=1000.0`.
    pub fn with_miter_limit(mut self, limit: f32) -> Result<Self, StrokeStyleError> {
        if !limit.is_finite() || !(1.0..=MAX_MITER_LIMIT).contains(&limit) {
            return Err(StrokeStyleError::InvalidMiterLimit);
        }
        self.miter_limit = limit;
        Ok(self)
    }

    /// Applies an allocation-free bounded dash pattern.
    pub const fn with_dash_pattern(mut self, dash: StrokeDashPattern2d) -> Self {
        self.dash = Some(dash);
        self
    }

    /// Adds an outward-pointing triangular marker based at the first path point.
    pub const fn with_start_marker(mut self, marker: StrokeMarker2d) -> Self {
        self.start_marker = Some(marker);
        self
    }

    /// Adds an outward-pointing triangular marker based at the last path point.
    pub const fn with_end_marker(mut self, marker: StrokeMarker2d) -> Self {
        self.end_marker = Some(marker);
        self
    }

    /// Returns color and scalar width.
    pub const fn stroke(self) -> Stroke {
        self.stroke
    }

    /// Returns whether width is logical-screen or world-space.
    pub const fn width_mode(self) -> StrokeWidthMode2d {
        self.width_mode
    }

    /// Returns endpoint presentation.
    pub const fn cap(self) -> StrokeCap2d {
        self.cap
    }

    /// Returns connected-segment presentation.
    pub const fn join(self) -> StrokeJoin2d {
        self.join
    }

    /// Returns the bounded miter multiple.
    pub const fn miter_limit(self) -> f32 {
        self.miter_limit
    }

    /// Returns the optional bounded dash pattern.
    pub const fn dash_pattern(self) -> Option<StrokeDashPattern2d> {
        self.dash
    }

    /// Returns the optional first-endpoint marker.
    pub const fn start_marker(self) -> Option<StrokeMarker2d> {
        self.start_marker
    }

    /// Returns the optional last-endpoint marker.
    pub const fn end_marker(self) -> Option<StrokeMarker2d> {
        self.end_marker
    }

    fn is_valid(self) -> bool {
        self.stroke.is_valid()
            && self.miter_limit.is_finite()
            && (1.0..=MAX_MITER_LIMIT).contains(&self.miter_limit)
    }
}

/// Invalid bounded line-style configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrokeStyleError {
    /// Dash patterns require an even count from two through eight.
    InvalidDashElementCount,
    /// Every dash and gap length must be positive and finite.
    InvalidDashLength,
    /// Dash phase must be finite.
    InvalidDashPhase,
    /// Dash expansion limit must be in `1..=1_000_000`.
    InvalidDashExpansionLimit,
    /// Miter limit must be finite and in `1.0..=1000.0`.
    InvalidMiterLimit,
}

impl fmt::Display for StrokeStyleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded 2D stroke style: {self:?}")
    }
}

impl Error for StrokeStyleError {}

/// Fill, stroke, and shadow styling for solid primitives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeStyle {
    fill: Option<Fill>,
    stroke: Option<Stroke>,
    shadow: Option<Shadow>,
}

impl ShapeStyle {
    /// Builds explicit fill, stroke, and shadow choices.
    pub const fn new(fill: Option<Fill>, stroke: Option<Stroke>, shadow: Option<Shadow>) -> Self {
        Self {
            fill,
            stroke,
            shadow,
        }
    }

    /// Returns fill paint, if any.
    pub const fn fill(self) -> Option<Fill> {
        self.fill
    }
    /// Returns outline stroke, if any.
    pub const fn stroke(self) -> Option<Stroke> {
        self.stroke
    }
    /// Returns shadow styling, if any.
    pub const fn shadow(self) -> Option<Shadow> {
        self.shadow
    }
    /// Creates a style with fill color only.
    pub fn filled(color: Color) -> Self {
        Self::filled_with(Fill::Solid(color))
    }

    /// Creates a style with fill paint only.
    pub fn filled_with(fill: Fill) -> Self {
        Self {
            fill: Some(fill),
            stroke: None,
            shadow: None,
        }
    }

    /// Creates a style with stroke only.
    ///
    /// `width` is in logical screen pixels.
    pub fn stroked(width: f32, color: Color) -> Self {
        Self {
            fill: None,
            stroke: Some(Stroke { width, color }),
            shadow: None,
        }
    }

    /// Creates a style with both fill and stroke.
    ///
    /// `stroke_width` is in logical screen pixels.
    pub fn fill_stroke(fill: Color, stroke_width: f32, stroke: Color) -> Self {
        Self::fill_stroke_with(Fill::Solid(fill), stroke_width, stroke)
    }

    /// Creates a style with both fill paint and stroke.
    ///
    /// `stroke_width` is in logical screen pixels.
    pub fn fill_stroke_with(fill: Fill, stroke_width: f32, stroke: Color) -> Self {
        Self {
            fill: Some(fill),
            stroke: Some(Stroke {
                width: stroke_width,
                color: stroke,
            }),
            shadow: None,
        }
    }

    /// Adds a shadow to the style.
    pub fn with_shadow(mut self, shadow: Shadow) -> Self {
        self.shadow = Some(shadow);
        self
    }

    fn validate(self, primitive: ScenePrimitive) -> Result<(), SceneError> {
        if self.fill.is_none() && self.stroke.is_none() && self.shadow.is_none() {
            return Err(SceneError::MissingStyle(primitive));
        }
        if self.fill.is_some_and(|fill| !fill.is_valid()) {
            return Err(SceneError::InvalidFill(primitive));
        }
        if self.stroke.is_some_and(|stroke| !stroke.is_valid()) {
            return Err(SceneError::InvalidStroke(primitive));
        }
        if self.shadow.is_some_and(|shadow| !shadow.is_valid()) {
            return Err(SceneError::InvalidShadow(primitive));
        }
        Ok(())
    }
}

/// Paint used to fill a shape.
///
/// Gradient coordinates are in world units. Renderers may approximate gradients
/// per vertex until a more advanced shader path exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fill {
    /// Single color over the whole shape.
    Solid(Color),
    /// Color interpolation along a world-space axis.
    LinearGradient(LinearGradient),
    /// Color interpolation outward from a world-space center.
    RadialGradient(RadialGradient),
}

impl Fill {
    /// Returns the color at a world-space point.
    ///
    /// Degenerate gradients fall back to their end color rather than producing
    /// NaN.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    pub(crate) fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        match self {
            Self::Solid(color) => color,
            Self::LinearGradient(gradient) => {
                gradient.color_at_with_offset(world_base, world_offset)
            }
            Self::RadialGradient(gradient) => {
                gradient.color_at_with_offset(world_base, world_offset)
            }
        }
    }

    fn is_valid(self) -> bool {
        match self {
            Self::Solid(color) => color.is_normalized(),
            Self::LinearGradient(gradient) => gradient.is_valid(),
            Self::RadialGradient(gradient) => gradient.is_valid(),
        }
    }
}

impl From<Color> for Fill {
    fn from(color: Color) -> Self {
        Self::Solid(color)
    }
}

/// Linear color gradient in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearGradient {
    start: Vec2,
    end: Vec2,
    start_color: Color,
    end_color: Color,
}

impl LinearGradient {
    /// Builds a linear gradient between two world-space points.
    pub fn new(start: Vec2, end: Vec2, start_color: Color, end_color: Color) -> Self {
        Self {
            start,
            end,
            start_color,
            end_color,
        }
    }

    /// Returns the world-space gradient start.
    pub const fn start(self) -> Vec2 {
        self.start
    }
    /// Returns the world-space gradient end.
    pub const fn end(self) -> Vec2 {
        self.end
    }
    /// Returns the color at the start.
    pub const fn start_color(self) -> Color {
        self.start_color
    }
    /// Returns the color at the end.
    pub const fn end_color(self) -> Color {
        self.end_color
    }

    /// Samples the gradient at a world-space point.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        if !self.is_valid() || !world_base.is_finite() || !world_offset.is_finite() {
            return self.end_color.clamp();
        }

        let axis_x = f64::from(self.end.x) - f64::from(self.start.x);
        let axis_y = f64::from(self.end.y) - f64::from(self.start.y);
        let relative_x =
            f64::from(world_base.x) - f64::from(self.start.x) + f64::from(world_offset.x);
        let relative_y =
            f64::from(world_base.y) - f64::from(self.start.y) + f64::from(world_offset.y);
        let axis_length_squared = axis_x * axis_x + axis_y * axis_y;
        let amount = ((relative_x * axis_x + relative_y * axis_y) / axis_length_squared)
            .clamp(0.0, 1.0) as f32;
        self.start_color.interpolate(self.end_color, amount)
    }

    fn is_valid(self) -> bool {
        self.start.is_finite()
            && self.end.is_finite()
            && self.start_color.is_normalized()
            && self.end_color.is_normalized()
            && self.start != self.end
    }
}

/// Radial color gradient in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialGradient {
    /// Center point in world coordinates.
    center: Vec2,
    /// Radius where `inner_color` is reached, in world units.
    inner_radius: f32,
    /// Radius where `outer_color` is reached, in world units.
    outer_radius: f32,
    /// Color at or inside `inner_radius`.
    inner_color: Color,
    /// Color at or outside `outer_radius`.
    outer_color: Color,
}

impl RadialGradient {
    /// Builds a radial gradient in world coordinates.
    ///
    /// Reversed finite radii are normalized together with their colors so the
    /// color attached to each radius remains stable.
    pub fn new(
        center: Vec2,
        inner_radius: f32,
        outer_radius: f32,
        inner_color: Color,
        outer_color: Color,
    ) -> Self {
        if inner_radius.is_finite() && outer_radius.is_finite() && inner_radius > outer_radius {
            Self {
                center,
                inner_radius: outer_radius,
                outer_radius: inner_radius,
                inner_color: outer_color,
                outer_color: inner_color,
            }
        } else {
            Self {
                center,
                inner_radius,
                outer_radius,
                inner_color,
                outer_color,
            }
        }
    }

    /// Returns the world-space gradient center.
    pub const fn center(self) -> Vec2 {
        self.center
    }

    /// Returns the inner radius in world units.
    pub const fn inner_radius(self) -> f32 {
        self.inner_radius
    }

    /// Returns the outer radius in world units.
    pub const fn outer_radius(self) -> f32 {
        self.outer_radius
    }

    /// Returns the inner linear color.
    pub const fn inner_color(self) -> Color {
        self.inner_color
    }

    /// Returns the outer linear color.
    pub const fn outer_color(self) -> Color {
        self.outer_color
    }

    /// Samples the gradient at a world-space point.
    pub fn color_at(self, world: Vec2) -> Color {
        self.color_at_with_offset(world, Vec2::ZERO)
    }

    fn color_at_with_offset(self, world_base: Vec2, world_offset: Vec2) -> Color {
        if !self.is_valid() || !world_base.is_finite() || !world_offset.is_finite() {
            return self.outer_color.clamp();
        }

        let radius_range = self.outer_radius - self.inner_radius;
        if radius_range == 0.0 {
            return self.outer_color;
        }

        let horizontal =
            f64::from(world_base.x) - f64::from(self.center.x) + f64::from(world_offset.x);
        let vertical =
            f64::from(world_base.y) - f64::from(self.center.y) + f64::from(world_offset.y);
        let distance = horizontal.hypot(vertical);
        let amount = ((distance - f64::from(self.inner_radius)) / f64::from(radius_range))
            .clamp(0.0, 1.0) as f32;
        self.inner_color.interpolate(self.outer_color, amount)
    }

    fn is_valid(self) -> bool {
        self.center.is_finite()
            && self.inner_radius.is_finite()
            && self.outer_radius.is_finite()
            && self.inner_radius >= 0.0
            && self.outer_radius >= self.inner_radius
            && self.inner_color.is_normalized()
            && self.outer_color.is_normalized()
    }
}

/// Stroke color and width for lines and shape outlines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stroke {
    width: f32,
    color: Color,
}

impl Stroke {
    /// Builds a stroke used by shape outlines and legacy logical-width lines.
    ///
    /// [`StrokeStyle2d::world`] reuses the same scalar/color value while making
    /// the width's coordinate space explicit on the containing style.
    pub const fn new(width: f32, color: Color) -> Self {
        Self { width, color }
    }
    /// Returns scalar width in the coordinate space selected by its owner.
    pub const fn width(self) -> f32 {
        self.width
    }
    /// Returns the stroke color.
    pub const fn color(self) -> Color {
        self.color
    }
    fn is_valid(self) -> bool {
        self.width.is_finite() && self.width > 0.0 && self.color.is_normalized()
    }
}

/// Simple screen-space shadow used by the first renderer slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shadow {
    offset: LogicalScreenVector,
    spread: f32,
    color: Color,
}

impl Shadow {
    /// Builds a shadow from logical screen-space offset, spread, and color.
    pub fn new(offset: LogicalScreenVector, spread: f32, color: Color) -> Self {
        Self {
            offset,
            spread,
            color,
        }
    }

    /// Returns the logical-pixel shadow offset.
    pub const fn offset(self) -> LogicalScreenVector {
        self.offset
    }
    /// Returns extra logical-pixel spread.
    pub const fn spread(self) -> f32 {
        self.spread
    }
    /// Returns the shadow color.
    pub const fn color(self) -> Color {
        self.color
    }

    fn is_valid(self) -> bool {
        self.offset.is_finite()
            && self.spread.is_finite()
            && self.spread >= 0.0
            && (self.spread * 2.0).is_finite()
            && self.color.is_normalized()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::MAX_STROKE_DASH_SUBSEGMENTS;
    use crate::{
        Color, DrawCommand, Fill, Layer, LinearGradient, LogicalScreenPosition,
        LogicalScreenVector, RadialGradient, Rect, Scene, SceneBudget, SceneBudgetResource,
        SceneError, ScenePrimitive, ScreenClipRect, ShapeStyle, StrokeCap2d, StrokeDashPattern2d,
        StrokeJoin2d, StrokeMarker2d, StrokeStyle2d, StrokeStyleError, Vec2,
    };
    use crate::{LogicalPixels, WorldLength};

    const UNLIMITED: usize = usize::MAX;

    const fn budget_for(
        commands: usize,
        points: usize,
        vertices: usize,
        retained_bytes: usize,
        upload_bytes: usize,
        batches: usize,
    ) -> SceneBudget {
        SceneBudget::new(
            commands,
            points,
            vertices,
            retained_bytes,
            upload_bytes,
            batches,
        )
    }

    #[test]
    fn scene_collects_visual_commands_only() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands.len(), 1);
    }

    #[test]
    fn scene_keeps_commands_sorted_by_layer() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene.circle_on_layer(
            Layer::FOREGROUND,
            Vec2::ZERO,
            10.0,
            ShapeStyle::filled(Color::WHITE),
        );
        scene.line_on_layer(Layer::BACKGROUND, Vec2::ZERO, Vec2::X, 1.0, Color::WHITE);

        assert!(matches!(scene.commands[0].command, DrawCommand::Line(_)));
        assert!(matches!(scene.commands[1].command, DrawCommand::Circle(_)));
    }

    #[test]
    fn scene_budget_rejection_is_atomic_and_counted() {
        let budget = budget_for(1, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED);
        let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
        scene
            .try_circle(Vec2::new(1.0, 2.0), 3.0, ShapeStyle::filled(Color::WHITE))
            .unwrap();

        let before_order = scene.commands()[0].order();
        let result = scene.try_circle(Vec2::new(4.0, 5.0), 6.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(
            result,
            Err(SceneError::BudgetExceeded {
                resource: SceneBudgetResource::Commands,
                limit: 1,
                requested: 2,
            })
        );
        assert_eq!(scene.command_count(), 1);
        assert_eq!(scene.commands()[0].order(), before_order);
        assert_eq!(scene.statistics().requested_commands(), 2);
        assert_eq!(scene.statistics().accepted_commands(), 1);
        assert_eq!(scene.statistics().rejected_commands(), 1);
    }

    #[test]
    fn scene_budget_enforces_every_resource_at_insertion() {
        let mut reference = Scene::new(Color::BLACK).unwrap();
        reference
            .try_polyline(
                vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)],
                2.0,
                Color::WHITE,
            )
            .unwrap();
        let usage = reference.statistics();
        let cases = [
            (
                SceneBudgetResource::Points,
                budget_for(
                    UNLIMITED,
                    usage.retained_points() - 1,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                ),
                usage.retained_points(),
            ),
            (
                SceneBudgetResource::TessellatedVertices,
                budget_for(
                    UNLIMITED,
                    UNLIMITED,
                    usage.estimated_tessellated_vertices() - 1,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                ),
                usage.estimated_tessellated_vertices(),
            ),
            (
                SceneBudgetResource::RetainedBytes,
                budget_for(
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    usage.retained_bytes() - 1,
                    UNLIMITED,
                    UNLIMITED,
                ),
                usage.retained_bytes(),
            ),
            (
                SceneBudgetResource::UploadBytes,
                budget_for(
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    usage.estimated_upload_bytes() - 1,
                    UNLIMITED,
                ),
                usage.estimated_upload_bytes(),
            ),
            (
                SceneBudgetResource::DrawBatches,
                budget_for(
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    UNLIMITED,
                    usage.estimated_draw_batches() - 1,
                ),
                usage.estimated_draw_batches(),
            ),
        ];

        for (resource, budget, requested) in cases {
            let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
            let result = scene.try_polyline(
                vec![Vec2::ZERO, Vec2::X, Vec2::new(1.0, 1.0)],
                2.0,
                Color::WHITE,
            );
            assert_eq!(
                result,
                Err(SceneError::BudgetExceeded {
                    resource,
                    limit: requested - 1,
                    requested,
                })
            );
            assert_eq!(scene.command_count(), 0);
            assert_eq!(scene.statistics().accepted_commands(), 0);
            assert_eq!(scene.statistics().rejected_commands(), 1);
        }
    }

    #[test]
    fn exact_scene_budget_is_accepted_and_clear_preserves_limits() {
        let mut reference = Scene::new(Color::BLACK).unwrap();
        reference
            .try_line(Vec2::ZERO, Vec2::X, 2.0, Color::WHITE)
            .unwrap();
        let usage = reference.statistics();
        let budget = budget_for(
            usage.accepted_commands(),
            usage.retained_points(),
            usage.estimated_tessellated_vertices(),
            usage.retained_bytes(),
            usage.estimated_upload_bytes(),
            usage.estimated_draw_batches(),
        );
        let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();

        scene
            .try_line(Vec2::ZERO, Vec2::X, 2.0, Color::WHITE)
            .unwrap();
        assert_eq!(scene.statistics(), usage);
        scene.clear();

        assert_eq!(scene.budget(), Some(budget));
        assert_eq!(scene.statistics(), Default::default());
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn batch_insertion_sorts_once_and_is_atomic_on_budget_failure() {
        let budget = budget_for(4, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED, UNLIMITED);
        let mut scene = Scene::with_budget(Color::BLACK, budget).unwrap();
        scene
            .try_extend_to_layers([
                (
                    Layer::FOREGROUND,
                    DrawCommand::circle(Vec2::new(1.0, 0.0), 1.0, ShapeStyle::filled(Color::WHITE))
                        .unwrap(),
                ),
                (
                    Layer::BACKGROUND,
                    DrawCommand::line(Vec2::ZERO, Vec2::X, 1.0, Color::WHITE).unwrap(),
                ),
                (
                    Layer::FOREGROUND,
                    DrawCommand::circle(Vec2::new(2.0, 0.0), 1.0, ShapeStyle::filled(Color::WHITE))
                        .unwrap(),
                ),
            ])
            .unwrap();

        assert_eq!(scene.command_count(), 3);
        assert_eq!(scene.commands()[0].layer(), Layer::BACKGROUND);
        assert_eq!(scene.commands()[1].order(), 0);
        assert_eq!(scene.commands()[2].order(), 2);

        let result = scene.try_extend_to_layers([
            (
                Layer::DEFAULT,
                DrawCommand::circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)).unwrap(),
            ),
            (
                Layer::DEFAULT,
                DrawCommand::circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE)).unwrap(),
            ),
        ]);
        assert_eq!(
            result,
            Err(SceneError::BudgetExceeded {
                resource: SceneBudgetResource::Commands,
                limit: 4,
                requested: 5,
            })
        );
        assert_eq!(scene.command_count(), 3);
        assert_eq!(scene.statistics().requested_commands(), 5);
        assert_eq!(scene.statistics().accepted_commands(), 3);
        assert_eq!(scene.statistics().rejected_commands(), 2);
    }

    #[test]
    fn linear_gradient_samples_along_axis() {
        let gradient = LinearGradient::new(Vec2::ZERO, Vec2::X, Color::BLACK, Color::WHITE);

        let sampled = Fill::LinearGradient(gradient).color_at(Vec2::new(0.5, 4.0));

        assert!((sampled.red() - 0.5).abs() < 0.001);
        assert!((sampled.green() - 0.5).abs() < 0.001);
        assert!((sampled.blue() - 0.5).abs() < 0.001);
    }

    #[test]
    fn extreme_gradients_sample_without_f32_overflow() {
        let gradient = LinearGradient::new(
            Vec2::new(-f32::MAX, 0.0),
            Vec2::new(f32::MAX, 0.0),
            Color::BLACK,
            Color::WHITE,
        );
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert!(
            scene
                .try_circle(
                    Vec2::ZERO,
                    1.0,
                    ShapeStyle::filled_with(Fill::LinearGradient(gradient)),
                )
                .is_ok()
        );
        assert_eq!(
            gradient.color_at(Vec2::ZERO),
            Color::rgba(0.5, 0.5, 0.5, 1.0)
        );

        let radial = RadialGradient::new(
            Vec2::splat(f32::MAX),
            0.0,
            f32::MAX,
            Color::BLACK,
            Color::WHITE,
        );
        assert_eq!(radial.color_at(Vec2::splat(-f32::MAX)), Color::WHITE);
    }

    #[test]
    fn overflowing_circle_and_shadow_are_rejected_at_scene_boundary() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert_eq!(
            scene.try_circle(
                Vec2::splat(f32::MAX),
                f32::MAX,
                ShapeStyle::filled(Color::WHITE),
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle::filled(Color::WHITE).with_shadow(super::Shadow::new(
                    LogicalScreenVector::new(0.0, 0.0),
                    f32::MAX,
                    Color::WHITE,
                )),
            ),
            Err(SceneError::InvalidShadow(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(
                Vec2::new(-f32::MAX, 0.0),
                Vec2::new(f32::MAX, 0.0),
                1.0,
                Color::WHITE,
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Line))
        );
        assert_eq!(
            scene.try_rect(
                Rect::new(Vec2::new(-f32::MAX, 0.0), Vec2::new(f32::MAX, 1.0),),
                0.0,
                ShapeStyle::filled(Color::WHITE),
            ),
            Err(SceneError::NonFiniteGeometry(ScenePrimitive::Rect))
        );
    }

    #[test]
    fn commands_capture_temporary_screen_clip() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let outer = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(10.0, 20.0),
            LogicalScreenVector::new(100.0, 80.0),
        )
        .unwrap();
        let inner = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(50.0, 0.0),
            LogicalScreenVector::new(100.0, 50.0),
        )
        .unwrap();

        scene
            .with_screen_clip(outer, |scene| {
                scene
                    .with_screen_clip(inner, |scene| {
                        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));
                    })
                    .unwrap();
            })
            .unwrap();
        scene.circle(Vec2::ZERO, 10.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(
            scene.commands[0].screen_clip,
            Some(
                ScreenClipRect::new(
                    LogicalScreenPosition::new(50.0, 20.0),
                    LogicalScreenPosition::new(110.0, 50.0),
                )
                .unwrap(),
            )
        );
        assert_eq!(scene.commands[1].screen_clip, None);
    }

    #[test]
    fn scene_rejects_invalid_primitives_before_ordering() {
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert!(!scene.circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)));
        assert!(!scene.line(Vec2::ZERO, Vec2::X, f32::NAN, Color::WHITE));
        assert!(!scene.polyline(vec![Vec2::ZERO], 1.0, Color::WHITE));
        assert_eq!(scene.command_count(), 0);

        assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
        assert_eq!(scene.commands()[0].order(), 0);
    }

    #[test]
    fn bounded_stroke_configuration_rejects_invalid_patterns_and_miters() {
        assert_eq!(
            StrokeDashPattern2d::new(&[1.0], 0.0, 4),
            Err(StrokeStyleError::InvalidDashElementCount)
        );
        assert_eq!(
            StrokeDashPattern2d::new(&[1.0, 0.0], 0.0, 4),
            Err(StrokeStyleError::InvalidDashLength)
        );
        assert_eq!(
            StrokeDashPattern2d::new(&[1.0, 1.0], f32::NAN, 4),
            Err(StrokeStyleError::InvalidDashPhase)
        );
        assert_eq!(
            StrokeDashPattern2d::new(&[1.0, 1.0], 0.0, 0),
            Err(StrokeStyleError::InvalidDashExpansionLimit)
        );
        assert_eq!(
            StrokeDashPattern2d::new(&[1.0, 1.0], 0.0, MAX_STROKE_DASH_SUBSEGMENTS + 1,),
            Err(StrokeStyleError::InvalidDashExpansionLimit)
        );
        assert_eq!(
            StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
                .with_miter_limit(0.5),
            Err(StrokeStyleError::InvalidMiterLimit)
        );
    }

    #[test]
    fn dash_boundaries_must_be_representable_at_the_source_coordinate_scale() {
        let dash =
            StrokeDashPattern2d::new(&[f32::MIN_POSITIVE, f32::MIN_POSITIVE], 0.0, 8).unwrap();
        let style = StrokeStyle2d::logical(LogicalPixels::new(1.0).unwrap(), Color::WHITE)
            .with_dash_pattern(dash);

        assert!(matches!(
            DrawCommand::styled_line(Vec2::ZERO, Vec2::new(f32::MAX, 0.0), style),
            Err(SceneError::UnrepresentableStrokePattern(
                ScenePrimitive::Line,
            ))
        ));
    }

    #[test]
    fn dash_expansion_is_rejected_atomically_at_the_scene_boundary() {
        let dash = StrokeDashPattern2d::new(&[2.0, 2.0], 0.0, 2).unwrap();
        let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
            .with_cap(StrokeCap2d::Butt)
            .with_dash_pattern(dash);
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert_eq!(
            scene.try_styled_line(Vec2::ZERO, Vec2::new(10.0, 0.0), style),
            Err(SceneError::StrokeExpansionLimitExceeded {
                primitive: ScenePrimitive::Line,
                limit: 2,
                required: 3,
            })
        );
        assert_eq!(scene.command_count(), 0);
        assert_eq!(scene.statistics().requested_commands(), 1);
        assert_eq!(scene.statistics().rejected_commands(), 1);
    }

    #[test]
    fn scene_statistics_group_requested_accepted_and_rejected_primitives() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .try_circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE))
            .unwrap();
        assert_eq!(
            scene.try_line(Vec2::ZERO, Vec2::ZERO, 1.0, Color::WHITE),
            Err(SceneError::DegenerateGeometry(ScenePrimitive::Line))
        );
        scene
            .try_polyline(vec![Vec2::ZERO, Vec2::X], 1.0, Color::WHITE)
            .unwrap();

        let statistics = scene.statistics();
        assert_eq!(statistics.requested_by_primitive().circles(), 1);
        assert_eq!(statistics.requested_by_primitive().lines(), 1);
        assert_eq!(statistics.requested_by_primitive().polylines(), 1);
        assert_eq!(statistics.requested_by_primitive().total(), 3);
        assert_eq!(statistics.accepted_by_primitive().circles(), 1);
        assert_eq!(statistics.accepted_by_primitive().polylines(), 1);
        assert_eq!(statistics.accepted_by_primitive().total(), 2);
        assert_eq!(statistics.rejected_by_primitive().lines(), 1);
        assert_eq!(statistics.rejected_by_primitive().total(), 1);
    }

    #[test]
    fn rich_stroke_values_are_reusable_and_world_overflow_is_rejected() {
        let marker = StrokeMarker2d::arrow(
            LogicalPixels::new(8.0).unwrap(),
            LogicalPixels::new(6.0).unwrap(),
        );
        let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE)
            .with_cap(StrokeCap2d::Square)
            .with_join(StrokeJoin2d::Bevel)
            .with_start_marker(marker)
            .with_end_marker(marker);
        let command =
            DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::new(2.0, 1.0)], style)
                .unwrap();
        assert_eq!(command.estimated_tessellated_vertices(), 24);

        let unsafe_world = StrokeStyle2d::world(WorldLength::new(f32::MAX).unwrap(), Color::WHITE);
        assert!(matches!(
            DrawCommand::styled_line(Vec2::ZERO, Vec2::X, unsafe_world),
            Err(SceneError::InvalidStroke(ScenePrimitive::Line))
        ));

        let unsafe_logical =
            StrokeStyle2d::logical(LogicalPixels::new(f32::MAX).unwrap(), Color::WHITE);
        assert!(matches!(
            DrawCommand::styled_line(Vec2::ZERO, Vec2::X, unsafe_logical),
            Err(SceneError::InvalidStroke(ScenePrimitive::Line))
        ));
    }

    #[test]
    fn polyline_rejects_zero_segments_and_numerically_reversed_turns() {
        let style = StrokeStyle2d::logical(LogicalPixels::new(2.0).unwrap(), Color::WHITE);
        assert!(matches!(
            DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::ZERO, Vec2::X], style),
            Err(SceneError::DegenerateGeometry(ScenePrimitive::Polyline))
        ));
        assert!(matches!(
            DrawCommand::styled_polyline(vec![Vec2::ZERO, Vec2::X, Vec2::ZERO], style),
            Err(SceneError::DegenerateStrokeTurn {
                primitive: ScenePrimitive::Polyline,
                vertex_index: 1,
            })
        ));
        assert!(matches!(
            DrawCommand::styled_polyline(
                vec![Vec2::ZERO, Vec2::X, Vec2::new(0.0, 0.000_000_1)],
                style,
            ),
            Err(SceneError::DegenerateStrokeTurn {
                primitive: ScenePrimitive::Polyline,
                vertex_index: 1,
            })
        ));
    }

    #[test]
    fn background_and_clip_are_validated_when_created() {
        assert!(matches!(
            Scene::new(Color::rgba(f32::NAN, 0.0, 0.0, 1.0)),
            Err(SceneError::InvalidBackground)
        ));
        assert!(matches!(
            Scene::new(Color::rgb(1.01, 0.0, 0.0)),
            Err(SceneError::InvalidBackground)
        ));
        assert_eq!(
            ScreenClipRect::new(
                LogicalScreenPosition::new(0.0, 0.0),
                LogicalScreenPosition::new(0.0, 10.0),
            ),
            Err(SceneError::InvalidScreenClip)
        );
        assert_eq!(
            ScreenClipRect::new(
                LogicalScreenPosition::new(f32::NAN, 0.0),
                LogicalScreenPosition::new(10.0, 10.0),
            ),
            Err(SceneError::InvalidScreenClip)
        );
    }

    #[test]
    fn render_bound_scene_colors_must_be_normalized() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle::filled(Color::rgb(2.0, 0.0, 0.0)),
            ),
            Err(SceneError::InvalidFill(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(Vec2::ZERO, Vec2::X, 1.0, Color::rgba(0.0, 0.0, 0.0, -0.01),),
            Err(SceneError::InvalidStroke(ScenePrimitive::Line))
        );
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn depth_does_not_reorder_commands_within_a_layer() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        scene
            .with_depth(4.0, |scene| {
                scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));
            })
            .unwrap();
        scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands()[0].depth(), 4.0);
        assert_eq!(scene.commands()[1].depth(), 0.0);
    }

    #[test]
    fn structured_scene_errors_preserve_rejection_reason() {
        let mut scene = Scene::new(Color::BLACK).unwrap();

        assert_eq!(
            scene.try_circle(Vec2::ZERO, -1.0, ShapeStyle::filled(Color::WHITE)),
            Err(SceneError::InvalidDimension(ScenePrimitive::Circle))
        );
        assert_eq!(
            scene.try_line(Vec2::ZERO, Vec2::ZERO, 1.0, Color::WHITE),
            Err(SceneError::DegenerateGeometry(ScenePrimitive::Line))
        );
        assert_eq!(
            scene.try_circle(
                Vec2::ZERO,
                1.0,
                ShapeStyle {
                    fill: None,
                    stroke: None,
                    shadow: None,
                },
            ),
            Err(SceneError::MissingStyle(ScenePrimitive::Circle))
        );
        assert_eq!(scene.command_count(), 0);
    }

    #[test]
    fn commands_capture_depth_and_restore_it_after_unwind() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = scene.with_depth(4.5, |scene| {
                scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE));
                panic!("intentional depth unwind");
            });
        }));
        assert!(panic_result.is_err());

        scene.circle(Vec2::X, 1.0, ShapeStyle::filled(Color::WHITE));

        assert_eq!(scene.commands()[0].depth(), 4.5);
        assert_eq!(scene.commands()[1].depth(), 0.0);
        assert_eq!(
            scene.with_depth(f32::NAN, |_| {}),
            Err(SceneError::NonFiniteDepth)
        );
    }

    #[test]
    fn gradients_sanitize_invalid_and_reversed_configuration() {
        let invalid_linear = LinearGradient::new(
            Vec2::new(f32::NAN, 0.0),
            Vec2::X,
            Color::BLACK,
            Color::WHITE,
        );
        assert_eq!(invalid_linear.color_at(Vec2::ZERO), Color::WHITE);

        let reversed = RadialGradient::new(Vec2::ZERO, 10.0, 5.0, Color::WHITE, Color::BLACK);
        assert_eq!(reversed.inner_radius(), 5.0);
        assert_eq!(reversed.outer_radius(), 10.0);
        assert_eq!(reversed.color_at(Vec2::ZERO), Color::BLACK);
    }

    #[test]
    fn tiny_nonzero_radial_range_preserves_both_color_endpoints() {
        let gradient =
            RadialGradient::new(Vec2::ZERO, 0.0, 0.000_000_01, Color::BLACK, Color::WHITE);

        assert_eq!(gradient.color_at(Vec2::ZERO), Color::BLACK);
        assert_eq!(
            gradient.color_at(Vec2::new(0.000_000_01, 0.0)),
            Color::WHITE
        );
    }

    #[test]
    fn temporary_screen_clip_restores_state_after_unwind() {
        let mut scene = Scene::new(Color::BLACK).unwrap();
        let clip = ScreenClipRect::from_min_size(
            LogicalScreenPosition::new(0.0, 0.0),
            LogicalScreenVector::new(100.0, 100.0),
        )
        .unwrap();

        let panic_result = catch_unwind(AssertUnwindSafe(|| {
            let _ = scene.with_screen_clip(clip, |_| panic!("intentional test panic"));
        }));
        assert!(panic_result.is_err());

        assert!(scene.circle(Vec2::ZERO, 1.0, ShapeStyle::filled(Color::WHITE)));
        assert_eq!(scene.commands()[0].screen_clip(), None);
    }
}
