//! Dynamic mesh ownership, updates and normal-stream contracts.

use super::*;
use crate::renderer::mesh3d::*;
use crate::renderer::*;

mod resources;

pub(in crate::renderer::mesh3d) use resources::{
    assert_gpu_dynamic_contract, assert_gpu_dynamic_recovery,
};
