//! Test-only GPU transport. Pixel interpretation and expected values stay in fixtures.

use std::{sync::mpsc, time::Duration};

/// Copies raw mapped bytes without interpreting formats, channels or row padding.
/// The caller owns copy encoding/submission and chooses its existing timeout.
pub(super) fn read_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    timeout: Duration,
) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).expect("GPU readback callback receiver");
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(timeout),
        })
        .expect("GPU readback polling");
    receiver
        .recv_timeout(timeout)
        .expect("GPU readback callback timed out")
        .expect("GPU buffer mapping failed");
    let bytes = slice.get_mapped_range().expect("GPU mapped bytes").to_vec();
    buffer.unmap();
    bytes
}
