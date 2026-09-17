//! Clipping, native surfaces, culling and bounded preflight contracts.

use super::*;
use crate::renderer::mesh3d::*;
use crate::renderer::*;

mod boundary_identity;
mod clipped_rendering;
mod clipping;
mod inside_classification;
mod native_edges;
mod native_policy;
mod offscreen_gpu;
mod scratch;
mod source_cache_gpu;
mod source_cache_identity;

pub(in crate::renderer::mesh3d) use clipped_rendering::{
    assert_gpu_clipped_recovery, assert_gpu_surface_contract, read_pixels as test_read_pixels,
    target as test_target,
};
pub(in crate::renderer::mesh3d) use native_edges::assert_gpu_native_edge_validation;
pub(in crate::renderer::mesh3d) use native_policy::assert_gpu_native_surface_policy;
pub(in crate::renderer::mesh3d) use offscreen_gpu::{
    assert_gpu_offscreen_culling, assert_gpu_offscreen_culling_recovery,
};
pub(in crate::renderer::mesh3d) use scratch::assert_gpu_preflight_scratch;
pub(in crate::renderer::mesh3d) use source_cache_gpu::assert_gpu_source_cache_contract;
