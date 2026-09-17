//! Frozen dev.6 full-filter oracle, independent of production mip regeneration.
//!
//! Keep downsample, linear_to_srgb8 and linear_channels unchanged when modifying
//! the production filter. Exact accumulation and rounding are the compatibility
//! reference for both baseline and candidate benchmark executables.

pub(super) struct Level {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u8>,
}

pub(super) fn rebuild(width: u32, height: u32, base: Vec<u8>) -> Vec<Level> {
    assert!(width > 0 && height > 0);
    assert_eq!(base.len(), width as usize * height as usize * 4);
    let mut levels = vec![Level {
        width,
        height,
        pixels: base,
    }];
    let linear_channels = linear_channels();
    while levels.last().unwrap().width > 1 || levels.last().unwrap().height > 1 {
        let source = levels.last().unwrap();
        let width = (source.width / 2).max(1);
        let height = (source.height / 2).max(1);
        let mut pixels = vec![0; width as usize * height as usize * 4];
        downsample(
            &source.pixels,
            source.width,
            source.height,
            &mut pixels,
            width,
            height,
            &linear_channels,
        );
        levels.push(Level {
            width,
            height,
            pixels,
        });
    }
    levels
}

#[allow(clippy::too_many_arguments)]
fn downsample(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    destination: &mut [u8],
    width: u32,
    height: u32,
    linear_channels: &[f64; 256],
) {
    // Integer-halving dimensions still need the entire source extent. Using a
    // fractional footprint includes the last row/column of odd-sized images.
    let horizontal_ratio = f64::from(source_width) / f64::from(width);
    let vertical_ratio = f64::from(source_height) / f64::from(height);
    for y in 0..height {
        let top = f64::from(y) * vertical_ratio;
        let bottom = (f64::from(y) + 1.0) * vertical_ratio;
        for x in 0..width {
            let left = f64::from(x) * horizontal_ratio;
            let right = (f64::from(x) + 1.0) * horizontal_ratio;
            let mut channels = [0.0; 3];
            let mut alpha_weight = 0.0;
            let mut total_weight = 0.0;
            for source_y in (top.floor() as u32)..(bottom.ceil() as u32).min(source_height) {
                let vertical_weight =
                    bottom.min(f64::from(source_y) + 1.0) - top.max(f64::from(source_y));
                for source_x in (left.floor() as u32)..(right.ceil() as u32).min(source_width) {
                    let horizontal_weight =
                        right.min(f64::from(source_x) + 1.0) - left.max(f64::from(source_x));
                    let weight = vertical_weight * horizontal_weight;
                    let index = (source_y as usize * source_width as usize + source_x as usize) * 4;
                    let alpha = f64::from(source[index + 3]) / 255.0;
                    for channel in 0..3 {
                        channels[channel] +=
                            linear_channels[source[index + channel] as usize] * alpha * weight;
                    }
                    alpha_weight += alpha * weight;
                    total_weight += weight;
                }
            }
            let index = (y as usize * width as usize + x as usize) * 4;
            let alpha = (alpha_weight / total_weight * 255.0).round() as u8;
            if alpha == 0 {
                destination[index..index + 4].fill(0);
                continue;
            }
            for channel in 0..3 {
                destination[index + channel] = linear_to_srgb8(channels[channel] / alpha_weight);
            }
            destination[index + 3] = alpha;
        }
    }
}

fn linear_to_srgb8(linear: f64) -> u8 {
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn linear_channels() -> [f64; 256] {
    std::array::from_fn(|channel| {
        let encoded = channel as f64 / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    })
}
