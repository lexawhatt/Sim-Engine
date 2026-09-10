use super::*;

#[test]
fn preparation_is_optional_aligned_and_bounded() {
    let mut batch = UniformUploadBatch::default();
    assert!(!batch.prepare(128, 31, 128, 128));
    assert!(!batch.prepare(0, 0, 0, 0));
    assert!(!batch.prepare(128, 32, 0, 128));
    assert!(!batch.prepare(128, 32, 127, 128));
    assert!(!batch.prepare(128, 32, 128, 127));
    assert!(!batch.prepare(127, 32, 128, 128));
    assert!(!batch.prepare(129, 32, 132, 132));
    assert_eq!(batch.cpu_bytes(), 0);
    assert!(batch.prepare(128, 32, 128, 128));
    assert_eq!(
        batch.allocation_bytes(),
        0,
        "prepare must not allocate GPU memory"
    );
    assert_eq!(batch.bytes.len(), 0);
    assert_eq!(batch.copies.len(), 0);
}

#[test]
fn stages_are_disjoint_exact_and_do_not_allocate() {
    let mut batch = UniformUploadBatch::default();
    assert!(!batch.stage(0, &[1; 4]));
    assert!(batch.prepare(128, 32, 128, 128));
    let capacities = (batch.bytes.capacity(), batch.copies.capacity());
    for index in 0..32 {
        // Sparse destination indices are not bounded by the dirty-copy count.
        assert!(batch.stage(index * 7, &[index as u8; 4]));
    }
    assert!(!batch.stage(224, &[9; 4]));
    assert_eq!(
        capacities,
        (batch.bytes.capacity(), batch.copies.capacity())
    );
    for (index, copy) in batch.copies.iter().enumerate() {
        assert_eq!(copy.binding_index, index * 7);
        assert_eq!(copy.source_offset, (index * 4) as u64);
        assert_eq!(copy.size, 4);
        assert_eq!(&batch.bytes[index * 4..index * 4 + 4], &[index as u8; 4]);
    }
    assert_eq!(
        batch.cpu_bytes(),
        capacities.0 + capacities.1 * std::mem::size_of::<PendingCopy>()
    );
}

#[test]
fn failed_stage_is_atomic_and_prepare_discards_prior_plan() {
    let mut batch = UniformUploadBatch::default();
    assert!(batch.prepare(128, 32, 128, 128));
    assert!(batch.stage(0, &[1; 64]));
    assert!(!batch.stage(1, &[]));
    assert!(!batch.stage(1, &[2; 3]));
    assert!(!batch.stage(1, &[2; 68]));
    assert_eq!(batch.bytes, vec![1; 64]);
    assert_eq!(batch.copies.len(), 1);
    assert!(batch.stage(1, &[2; 64]));
    assert!(!batch.prepare(128, 31, 128, 128));
    assert!(batch.bytes.is_empty());
    assert!(batch.copies.is_empty());
    assert!(!batch.stage(0, &[3; 4]));
}

#[test]
fn retained_cpu_capacities_grow_reuse_and_clear() {
    let mut batch = UniformUploadBatch::default();
    assert!(batch.prepare(128, 32, 1024, 1024));
    let small = batch.cpu_bytes();
    assert!(batch.prepare(1024, 64, 1024, 1024));
    let large = batch.cpu_bytes();
    assert!(large > small);
    assert!(batch.stage(3, &[4; 96]));
    batch.begin();
    assert_eq!(batch.cpu_bytes(), large);
    assert!(batch.bytes.is_empty());
    assert!(batch.copies.is_empty());
    assert!(!batch.stage(0, &[0; 4]));
    assert!(batch.prepare(128, 32, 1024, 1024));
    assert_eq!(
        batch.cpu_bytes(),
        large,
        "CPU spare capacity remains accounted until eviction"
    );
    assert!(batch.prepare(128, 32, 128, 128));
    assert!(batch.bytes.capacity() <= 128);
    assert!(batch.cpu_bytes() < large);
    batch.clear();
    assert_eq!(batch.cpu_bytes(), 0);
    assert_eq!(batch.allocation_bytes(), 0);
}

#[test]
fn impossible_capacity_and_integer_overflow_fall_back_without_panicking() {
    let mut batch = UniformUploadBatch::default();
    assert!(!batch.prepare(usize::MAX, usize::MAX, usize::MAX, u64::MAX));
    assert!(!batch.prepare(usize::MAX & !3, 32, usize::MAX, u64::MAX));
    assert!(!batch.prepare(usize::MAX & !3, usize::MAX / 4, usize::MAX, u64::MAX));
    assert!(!batch.active);
    assert!(batch.bytes.is_empty());
    assert!(batch.copies.is_empty());
    assert_eq!(batch.allocation_bytes(), 0);
}
