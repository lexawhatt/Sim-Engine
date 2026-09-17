use super::{
    FrameComposer, FrameComposerError, FrameItem, FramePassOptions, FrameReport, FrameSourceKind,
    FrameSourceStatistics, FrameStatistics, RetainedResourceAccounting, RetainedResourceKey,
    ScalarLutPlan, account_new_retained_resources, retained_arc_key, retained_key,
    scalar_lut_counts_with_inserted, validate_frame_budget,
};
use crate::renderer::{
    BlendMode, COLOR_MAP_LUT_SIZE, Camera2d, CameraUniform, Color, ColorMap, CompositeUniform,
    DynamicGpu, DynamicMesh2d, GlyphAtlas2d, GlyphRun2d, HeatmapUniform, Image2d, ImageBatch2d,
    ImageBatchPlacement, ImageSampling, ImageTexelRect, ImageUniform, ParticleField2d, ParticleGpu,
    PreparedDrawBatch, PreparedScene, PreparedScreenScene, Rect, RenderTarget2d,
    ScalarFieldSampling, ScalarFieldTexture, Scene, ScreenScene, Vertex, color_map_lut,
    prepared_scene_belongs_to, scalar_normalization_is_portable, scalar_value_range_extent,
};

impl<'frame> FrameComposer<'frame> {
    /// Adds a streaming world-space scene through its own camera.
    pub fn draw_scene(
        &mut self,
        scene: &'frame Scene,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        let statistics = scene.statistics();
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count(),
            vertex_count: statistics.estimated_tessellated_vertices(),
            streaming_vertex_count: statistics.estimated_tessellated_vertices(),
            reused_vertex_count: 0,
            upload_bytes: statistics
                .estimated_upload_bytes()
                .saturating_add(std::mem::size_of::<CameraUniform>()),
            streaming_upload_bytes: statistics.estimated_upload_bytes(),
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: statistics.estimated_draw_batches(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_key(scene, 1),
                scene.allocation_bytes(),
                0,
                0,
            )],
            work,
            FrameItem::Scene {
                scene,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a streaming logical-screen scene unaffected by world cameras.
    pub fn draw_screen_scene(
        &mut self,
        scene: &'frame ScreenScene,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        let statistics = scene.statistics();
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count(),
            vertex_count: statistics.estimated_tessellated_vertices(),
            streaming_vertex_count: statistics.estimated_tessellated_vertices(),
            reused_vertex_count: 0,
            upload_bytes: statistics
                .estimated_upload_bytes()
                .saturating_add(std::mem::size_of::<CameraUniform>()),
            streaming_upload_bytes: statistics.estimated_upload_bytes(),
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: statistics.estimated_draw_batches(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::StreamingScene),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_key(scene, 2),
                scene.allocation_bytes(),
                0,
                0,
            )],
            work,
            FrameItem::ScreenScene {
                scene,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds immutable prepared world-space geometry through its own camera.
    pub fn draw_prepared_scene(
        &mut self,
        scene: &'frame PreparedScene,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if !prepared_scene_belongs_to(&self.renderer.renderer_identity, &scene.renderer_identity) {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::PreparedScene,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.command_count,
            vertex_count: scene.vertex_count,
            streaming_vertex_count: 0,
            reused_vertex_count: scene.vertex_count,
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: scene.draw_batches.len(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::PreparedScene),
        };
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.vertices, 13),
                    scene
                        .vertices
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.vertex_buffer, 3),
                    0,
                    scene
                        .vertex_count
                        .max(1)
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(&scene.draw_batches, 14),
                    scene
                        .draw_batches
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
                    0,
                    0,
                ),
            ],
            work,
            FrameItem::Prepared {
                scene,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds immutable prepared logical-screen geometry.
    pub fn draw_prepared_screen_scene(
        &mut self,
        scene: &'frame PreparedScreenScene,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if !prepared_scene_belongs_to(
            &self.renderer.renderer_identity,
            &scene.scene.renderer_identity,
        ) {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::PreparedScene,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: scene.scene.command_count,
            vertex_count: scene.scene.vertex_count,
            streaming_vertex_count: 0,
            reused_vertex_count: scene.scene.vertex_count,
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: scene.scene.draw_batches.len(),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::PreparedScene),
        };
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.scene.vertices, 13),
                    scene
                        .scene
                        .vertices
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_arc_key(&scene.scene.vertex_buffer, 3),
                    0,
                    scene
                        .scene
                        .vertex_count
                        .max(1)
                        .saturating_mul(std::mem::size_of::<Vertex>()),
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(&scene.scene.draw_batches, 14),
                    scene
                        .scene
                        .draw_batches
                        .capacity()
                        .saturating_mul(std::mem::size_of::<PreparedDrawBatch>()),
                    0,
                    0,
                ),
            ],
            work,
            FrameItem::PreparedScreen {
                scene,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds retained dynamic triangles through their own world camera.
    ///
    /// Camera-dependent topology is validated by [`FrameComposer::present`],
    /// which rejects triangles crossing the full target clip volume or having
    /// an ambiguous projected orientation. Item viewport/scissor clipping is
    /// axis-aligned fragment clipping and remains supported.
    pub fn draw_dynamic_mesh(
        &mut self,
        mesh: &'frame DynamicMesh2d,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_dynamic_mesh(mesh).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::DynamicMesh,
            });
        }
        let work = FrameStatistics {
            pass_count: 1,
            command_count: usize::from(!mesh.vertices.is_empty()),
            vertex_count: mesh.vertices.len(),
            streaming_vertex_count: 0,
            reused_vertex_count: mesh.vertices.len(),
            upload_bytes: std::mem::size_of::<CameraUniform>(),
            streaming_upload_bytes: 0,
            texture_bytes: 0,
            retained_cpu_bytes: 0,
            retained_buffer_bytes: 0,
            draw_calls: usize::from(!mesh.vertices.is_empty()),
            source_counts: FrameSourceStatistics::single(FrameSourceKind::DynamicMesh),
        };
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&mesh.vertex_buffer, 4),
                mesh.recovery_memory_bytes(),
                mesh.vertex_capacity
                    .saturating_mul(std::mem::size_of::<DynamicGpu>()),
                0,
            )],
            work,
            FrameItem::Dynamic {
                mesh,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a budgeted instanced particle field through its own world camera.
    pub fn draw_particle_field(
        &mut self,
        field: &'frame mut ParticleField2d,
        camera: Camera2d,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_particle_field(field).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::ParticleField,
            });
        }
        let candidate_count = field
            .instances
            .len()
            .min(field.budget.max_visibility_checks_per_frame)
            .min(field.budget.instance_limit());
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&field.instance_buffer, 5),
                field.cpu_allocation_bytes(),
                field.gpu_allocation_bytes(),
                0,
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(!field.instances.is_empty()),
                vertex_count: candidate_count.saturating_mul(6),
                streaming_vertex_count: candidate_count.saturating_mul(6),
                reused_vertex_count: 0,
                upload_bytes: candidate_count
                    .saturating_mul(std::mem::size_of::<ParticleGpu>())
                    .saturating_add(std::mem::size_of::<CameraUniform>()),
                streaming_upload_bytes: candidate_count
                    .saturating_mul(std::mem::size_of::<ParticleGpu>()),
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(candidate_count > 0),
                source_counts: FrameSourceStatistics::single(FrameSourceKind::ParticleField),
            },
            FrameItem::Particle {
                field,
                camera,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a retained scalar heatmap scaled into the optional viewport.
    pub fn draw_scalar_field(
        &mut self,
        texture: &'frame ScalarFieldTexture,
        color_map: &'frame ColorMap,
        (minimum, maximum): (f32, f32),
        sampling: ScalarFieldSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self
            .renderer
            .validate_scalar_field_texture(texture)
            .is_err()
        {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::ScalarField,
            });
        }
        let value_extent = scalar_value_range_extent(minimum, maximum)
            .ok_or(FrameComposerError::InvalidValueRange { minimum, maximum })?;
        if !scalar_normalization_is_portable(texture, minimum, value_extent) {
            return Err(FrameComposerError::InvalidValueRange { minimum, maximum });
        }
        let color_map_bytes = COLOR_MAP_LUT_SIZE as usize * 4;
        let scalar_lut = ScalarLutPlan {
            sort_key: (options.order(), self.next_insertion),
            lut: color_map_lut(color_map),
        };
        self.scalar_luts
            .try_reserve(1)
            .map_err(|_| FrameComposerError::AllocationFailed {
                requested_bytes: std::mem::size_of::<ScalarLutPlan>(),
            })?;
        let insertion = self
            .scalar_luts
            .partition_point(|planned| planned.sort_key < scalar_lut.sort_key);
        let (lut_upload_count, lut_allocation_count) = scalar_lut_counts_with_inserted(
            &self.scalar_luts,
            insertion,
            &scalar_lut.lut,
            self.renderer
                .color_map_cache
                .as_ref()
                .map(|cached| &cached.lut),
        );
        let additional_lut_uploads = lut_upload_count.saturating_sub(self.scalar_lut_upload_count);
        let additional_lut_allocations =
            lut_allocation_count.saturating_sub(self.scalar_lut_allocation_count);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_key(&texture.texture, 6),
                    texture.recovery_memory_bytes(),
                    0,
                    texture.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(color_map, 15),
                    color_map.allocation_bytes(),
                    0,
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<HeatmapUniform>()
                    .saturating_add(additional_lut_uploads.saturating_mul(color_map_bytes)),
                streaming_upload_bytes: 0,
                texture_bytes: additional_lut_allocations.saturating_mul(color_map_bytes),
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::ScalarField),
            },
            FrameItem::Scalar {
                texture,
                color_map,
                minimum,
                maximum,
                value_extent,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )?;
        self.scalar_luts.insert(insertion, scalar_lut);
        self.scalar_lut_upload_count = lut_upload_count;
        self.scalar_lut_allocation_count = lut_allocation_count;
        Ok(())
    }

    /// Adds one image or atlas region scaled into the optional viewport.
    ///
    /// With no viewport the image covers the complete logical surface. `tint`
    /// is normalized straight linear RGBA and multiplies decoded image color.
    pub fn draw_image(
        &mut self,
        image: &'frame Image2d,
        source: Option<ImageTexelRect>,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image(image).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        if !tint.is_normalized() {
            return Err(FrameComposerError::InvalidTint);
        }
        let source = source.unwrap_or_else(|| image.full_rect());
        if !source.fits(image.width(), image.height()) {
            return Err(FrameComposerError::InvalidImageRegion);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&image.resource_identity, 7),
                image.recovery_memory_bytes(),
                0,
                image.gpu_allocation_bytes(),
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Image),
            },
            FrameItem::Image {
                image,
                source,
                tint,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds one image or atlas region on an axis-aligned world-space rectangle.
    ///
    /// The rectangle follows the supplied camera's zoom, rotation, and
    /// pseudo-depth projection. Its UV top edge maps to the rectangle's maximum
    /// world Y edge. The optional frame viewport selects the camera viewport and
    /// the item remains constrained by that viewport and clip.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_world_image(
        &mut self,
        image: &'frame Image2d,
        source: Option<ImageTexelRect>,
        rectangle: Rect,
        depth: f32,
        camera: Camera2d,
        tint: Color,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image(image).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        if !tint.is_normalized() {
            return Err(FrameComposerError::InvalidTint);
        }
        let source = source.unwrap_or_else(|| image.full_rect());
        if !source.fits(image.width(), image.height()) {
            return Err(FrameComposerError::InvalidImageRegion);
        }
        let rectangle = rectangle.normalized();
        if !rectangle.min().is_finite()
            || !rectangle.max().is_finite()
            || !rectangle.width().is_finite()
            || !rectangle.height().is_finite()
            || rectangle.width() <= 0.0
            || rectangle.height() <= 0.0
            || !depth.is_finite()
        {
            return Err(FrameComposerError::InvalidWorldImage);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&image.resource_identity, 7),
                image.recovery_memory_bytes(),
                0,
                image.gpu_allocation_bytes(),
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Image),
            },
            FrameItem::WorldImage {
                image,
                source,
                rectangle,
                depth,
                camera,
                tint,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds a retained atlas batch as one instanced draw when non-empty.
    /// Empty batches are valid frame items and emit no draw call.
    pub fn draw_image_batch(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        self.draw_image_batch_placed(
            image,
            batch,
            ImageBatchPlacement::default(),
            sampling,
            options,
        )
    }

    /// Draws a retained batch with independent logical translation and tint.
    /// The pass viewport and clip stay fixed while the content moves; negative
    /// offsets and partially offscreen content are supported without re-upload.
    pub fn draw_image_batch_placed(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        placement: ImageBatchPlacement,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_image_batch(image, batch).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Image,
            });
        }
        self.push_retained_image_batch(
            image,
            batch,
            placement,
            sampling,
            options,
            FrameSourceKind::Image,
        )
    }

    /// Adds one host-shaped retained glyph run as one instanced draw when non-empty.
    ///
    /// Positions are local logical pixels. Mixed font fallback is represented
    /// by multiple runs submitted at the same order; stable insertion order is
    /// preserved between them. An empty run is valid and emits no draw call.
    pub fn draw_glyph_run(
        &mut self,
        atlas: &'frame GlyphAtlas2d,
        run: &'frame GlyphRun2d,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        self.draw_glyph_run_placed(
            atlas,
            run,
            ImageBatchPlacement::default(),
            sampling,
            options,
        )
    }

    /// Draws a shared glyph layout with independent logical translation and tint.
    /// Glyph bearings and per-glyph tint are preserved; draw tint multiplies them.
    /// The item clip is not translated. Existing default calls are equivalent
    /// to zero translation and white tint.
    pub fn draw_glyph_run_placed(
        &mut self,
        atlas: &'frame GlyphAtlas2d,
        run: &'frame GlyphRun2d,
        placement: ImageBatchPlacement,
        sampling: ImageSampling,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_glyph_run(atlas, run).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::Glyph,
            });
        }
        let image = &atlas.image;
        let batch = &run.batch;
        let vertex_count = batch.sprite_count().saturating_mul(6);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&image.resource_identity, 7),
                    image.recovery_memory_bytes(),
                    0,
                    image.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(atlas, 9),
                    atlas
                        .recovery_memory_bytes()
                        .saturating_sub(image.recovery_memory_bytes()),
                    0,
                    0,
                ),
                RetainedResourceAccounting::new(
                    retained_key(run, 10),
                    run.recovery_memory_bytes(),
                    run.gpu_allocation_bytes(),
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(batch.sprite_count() > 0),
                vertex_count,
                streaming_vertex_count: 0,
                reused_vertex_count: vertex_count,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(batch.sprite_count() > 0),
                source_counts: FrameSourceStatistics::single(FrameSourceKind::Glyph),
            },
            FrameItem::ImageBatch {
                image,
                batch,
                placement,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    fn push_retained_image_batch(
        &mut self,
        image: &'frame Image2d,
        batch: &'frame ImageBatch2d,
        placement: ImageBatchPlacement,
        sampling: ImageSampling,
        options: FramePassOptions,
        source: FrameSourceKind,
    ) -> Result<(), FrameComposerError> {
        let vertex_count = batch.sprite_count().saturating_mul(6);
        self.push_accounted_item(
            &[
                RetainedResourceAccounting::new(
                    retained_arc_key(&image.resource_identity, 7),
                    image.recovery_memory_bytes(),
                    0,
                    image.gpu_allocation_bytes(),
                ),
                RetainedResourceAccounting::new(
                    retained_key(batch, 8),
                    batch.recovery_memory_bytes(),
                    batch.gpu_allocation_bytes(),
                    0,
                ),
            ],
            FrameStatistics {
                pass_count: 1,
                command_count: usize::from(batch.sprite_count() > 0),
                vertex_count,
                streaming_vertex_count: 0,
                reused_vertex_count: vertex_count,
                upload_bytes: std::mem::size_of::<ImageUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: usize::from(batch.sprite_count() > 0),
                source_counts: FrameSourceStatistics::single(source),
            },
            FrameItem::ImageBatch {
                image,
                batch,
                placement,
                sampling,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Adds an offscreen target, scaled into the optional logical viewport.
    pub fn draw_render_target(
        &mut self,
        target: &'frame RenderTarget2d,
        blend_mode: BlendMode,
        opacity: f32,
        options: FramePassOptions,
    ) -> Result<(), FrameComposerError> {
        if self.renderer.validate_render_target(target).is_err() {
            return Err(FrameComposerError::RendererMismatch {
                source: FrameSourceKind::RenderTarget,
            });
        }
        if !opacity.is_finite() || !(0.0..=1.0).contains(&opacity) {
            return Err(FrameComposerError::InvalidOpacity);
        }
        self.push_accounted_item(
            &[RetainedResourceAccounting::new(
                retained_arc_key(&target.resource_identity, 11),
                0,
                0,
                target.allocation_bytes,
            )],
            FrameStatistics {
                pass_count: 1,
                command_count: 1,
                vertex_count: 6,
                streaming_vertex_count: 0,
                reused_vertex_count: 0,
                upload_bytes: std::mem::size_of::<CompositeUniform>(),
                streaming_upload_bytes: 0,
                texture_bytes: 0,
                retained_cpu_bytes: 0,
                retained_buffer_bytes: 0,
                draw_calls: 1,
                source_counts: FrameSourceStatistics::single(FrameSourceKind::RenderTarget),
            },
            FrameItem::Target {
                target,
                blend_mode,
                opacity,
                options,
                insertion: self.next_insertion,
            },
        )
    }

    /// Returns conservative work accepted so far without presenting it.
    pub const fn planned_statistics(&self) -> FrameStatistics {
        self.planned
    }

    /// Validates, orders, encodes, submits, and presents the complete frame.
    pub fn present(self) -> Result<FrameReport, FrameComposerError> {
        super::present::present_frame(self)
    }

    fn push_item(
        &mut self,
        work: FrameStatistics,
        item: FrameItem<'frame>,
    ) -> Result<(), FrameComposerError> {
        let planned = self.planned.adding(work);
        validate_frame_budget(self.budget, planned)?;
        self.items
            .try_reserve(1)
            .map_err(|_| FrameComposerError::AllocationFailed {
                requested_bytes: std::mem::size_of::<FrameItem<'frame>>(),
            })?;
        self.items.push(item);
        self.planned = planned;
        self.next_insertion = self.next_insertion.saturating_add(1);
        Ok(())
    }

    fn push_accounted_item(
        &mut self,
        resources: &[RetainedResourceAccounting],
        mut work: FrameStatistics,
        item: FrameItem<'frame>,
    ) -> Result<(), FrameComposerError> {
        let (accounted_work, missing) =
            account_new_retained_resources(&self.retained_resources, resources, work);
        work = accounted_work;
        self.retained_resources.try_reserve(missing).map_err(|_| {
            FrameComposerError::AllocationFailed {
                requested_bytes: missing.saturating_mul(std::mem::size_of::<RetainedResourceKey>()),
            }
        })?;
        self.push_item(work, item)?;
        for resource in resources {
            if !self.retained_resources.contains(&resource.key) {
                self.retained_resources.push(resource.key);
            }
        }
        Ok(())
    }
}
