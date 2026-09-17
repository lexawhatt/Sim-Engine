//! Material, lifecycle and complete-byte texture-update contracts.

use super::*;
use crate::renderer::mesh3d::*;
use crate::renderer::*;

mod atomicity;
pub(super) mod full_mip_reference;
mod lifecycle;
mod materials;
mod regional_updates;
mod restoration;
mod update_benchmark;

pub(in crate::renderer::mesh3d) use lifecycle::{
    assert_gpu_texture_lifecycle, assert_gpu_texture_lifecycle_recovery,
};
pub(in crate::renderer::mesh3d) use materials::{
    assert_gpu_texture_contract, assert_gpu_textured_recovery,
};
pub(in crate::renderer::mesh3d) use regional_updates::{
    assert_gpu_partial_mip_recovery, assert_gpu_partial_mip_updates,
};
pub(in crate::renderer::mesh3d) use restoration::assert_gpu_restored_material_policy;
pub(in crate::renderer::mesh3d) use update_benchmark::run as benchmark_texture_updates;
