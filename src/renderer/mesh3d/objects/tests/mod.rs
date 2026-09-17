//! Scene accounting and presentation contracts.

use super::*;
use crate::renderer::mesh3d::*;
use crate::renderer::*;

mod accounting;
mod accounting_gpu;
mod presentation;

pub(in crate::renderer::mesh3d) use accounting_gpu::verify_scene_accounting;
