//! Retained 3D regression tests, grouped by the contract under test.

use super::*;
use crate::renderer::*;

mod edges;
mod resources;
mod transforms;

mod accounting_benchmark;
mod color_budget;
mod edge_upload;
mod frame_upload;
mod gpu_contract;
mod illumination;
mod instancing;
mod lifetime;
mod material_resource;
mod materials;
mod native_acceptance;
mod timing;
mod vertex_color;

pub(in crate::renderer) use gpu_contract::{
    assert_gpu_depth_contract, assert_gpu_scene_recovery_contract,
};
pub(in crate::renderer) use timing::assert_gpu_scene_timing_and_upload_contract;
