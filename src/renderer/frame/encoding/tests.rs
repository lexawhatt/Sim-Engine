use super::*;
use wgpu::util::DeviceExt;

mod gpu;
pub(in crate::renderer) use gpu::assert_gpu_encoding_contract;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Command {
    Pipeline(u8),
    Buffer(usize, u8),
    Scissor(ScissorRect),
    Draw(u8),
}

#[test]
fn setter_filter_preserves_each_draw_state_and_order() {
    let full = ScissorRect {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let clipped = ScissorRect {
        x: 7,
        y: 3,
        width: 29,
        height: 41,
    };
    let commands = [
        Command::Pipeline(1),
        Command::Buffer(0, 1),
        Command::Scissor(full),
        Command::Draw(0),
        Command::Pipeline(1),
        Command::Buffer(0, 1),
        Command::Scissor(full),
        Command::Draw(1),
        Command::Pipeline(2),
        Command::Scissor(clipped),
        Command::Draw(2),
        Command::Pipeline(3),
        Command::Buffer(0, 2),
        Command::Buffer(1, 3),
        Command::Scissor(full),
        Command::Draw(3),
        // A pipeline that does not use slot one must not erase its binding.
        Command::Pipeline(2),
        Command::Scissor(full),
        Command::Draw(4),
        Command::Pipeline(3),
        Command::Buffer(0, 2),
        Command::Buffer(1, 3),
        Command::Scissor(full),
        Command::Draw(5),
        Command::Pipeline(1),
        Command::Buffer(0, 1),
        Command::Scissor(clipped),
        Command::Draw(6),
    ];
    let filter = || {
        let mut buffers: [LastValue<u8>; 2] = std::array::from_fn(|_| LastValue::default());
        let mut scissor = LastValue::default();
        commands
            .into_iter()
            .filter(|command| match *command {
                Command::Buffer(slot, buffer) => buffers[slot].changed(buffer),
                Command::Scissor(rect) => scissor.changed(rect),
                _ => true,
            })
            .collect::<Vec<_>>()
    };
    let draws = |commands: &[Command]| {
        let mut pipeline = 0;
        let mut buffers = [None; 2];
        let mut scissor = None;
        commands
            .iter()
            .filter_map(|command| {
                match *command {
                    Command::Pipeline(value) => pipeline = value,
                    Command::Buffer(slot, value) => buffers[slot] = Some(value),
                    Command::Scissor(value) => scissor = Some(value),
                    Command::Draw(id) => return Some((id, pipeline, buffers, scissor)),
                }
                None
            })
            .collect::<Vec<_>>()
    };
    let reduced = filter();
    assert_eq!(draws(&commands), draws(&reduced));
    assert!(reduced.len() < commands.len());
    // Every new pass starts with empty cache, even if the command list is equal.
    assert_eq!(filter(), reduced);
    let mut cache = LastValue::default();
    assert!(cache.changed(full));
    assert!(!cache.changed(full));
    assert!(cache.changed(clipped));
    assert!(cache.changed(full));
}
