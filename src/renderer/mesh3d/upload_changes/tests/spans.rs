use super::*;

fn apply(span: &UploadSpan, mirror: &[u8], gpu: &mut [u8]) {
    if let Some(bytes) = span.range() {
        gpu[bytes.clone()].copy_from_slice(&mirror[bytes]);
    }
}

#[test]
fn unchanged_records_and_empty_ranges_require_no_upload() {
    let mut span = UploadSpan::default();
    let mut record = [1u32, 2, 3, 4];
    assert!(!span.replace(16, &mut record, [1, 2, 3, 4]));
    span.mark(12..12);
    assert_eq!(span.range(), None);
    assert_eq!(span.byte_count(), 0);
}

#[test]
fn separated_changes_merge_full_records_and_align_byte_ranges() {
    let mut span = UploadSpan::default();
    let mut records = [[0u32; 4]; 5];
    assert!(span.replace(64, &mut records[4], [1, 2, 3, 4]));
    assert!(span.replace(16, &mut records[1], [5, 6, 7, 8]));
    assert_eq!(span.range(), Some(16..80));
    assert_eq!(span.byte_count(), 64);
    let mut bytes = [0; 24];
    let mut byte_span = UploadSpan::default();
    assert!(byte_span.replace_bytes(9, &mut bytes[9..11], &[1, 2]));
    assert_eq!(byte_span.range(), Some(8..12));
    byte_span.mark(2..3);
    assert_eq!(byte_span.range(), Some(0..12));
    byte_span.mark(19..20);
    assert_eq!(byte_span.range(), Some(0..20));
}

#[test]
fn pod_comparison_preserves_signed_zero_and_nan_payloads() {
    let payload = f32::from_bits(0x7fc0_0042);
    let mut record = [0.0f32, payload, 1.0, 2.0];
    let mut unchanged = UploadSpan::default();
    assert!(!unchanged.replace(0, &mut record, [0.0, payload, 1.0, 2.0]));
    assert_eq!(unchanged.range(), None);
    let mut changed = UploadSpan::default();
    assert!(changed.replace(0, &mut record, [-0.0, payload, 1.0, 2.0]));
    assert_eq!(record[0].to_bits(), (-0.0f32).to_bits());
    assert_eq!(changed.range(), Some(0..16));
    let other_payload = f32::from_bits(0x7fc0_0043);
    let mut changed = UploadSpan::default();
    assert!(changed.replace(0, &mut record, [-0.0, other_payload, 1.0, 2.0]));
    assert_eq!(record[1].to_bits(), other_payload.to_bits());
}

#[test]
fn shrink_needs_no_write_but_newly_active_slots_need_residency() {
    let mut mirror = vec![7u8; 48];
    let mut gpu = vec![7u8; 48];
    mirror.truncate(16);
    let shrink = UploadSpan::default();
    assert_eq!(shrink.range(), None);
    apply(&shrink, &mirror, &mut gpu);
    let old_length = mirror.len();
    mirror.resize(48, 0);
    let mut regrowth = UploadSpan::default();
    // A zero record compares equal to newly zeroed staging but GPU still has 7.
    assert!(!regrowth.replace_bytes(16, &mut mirror[16..32], &[0; 16]));
    regrowth.mark(old_length..mirror.len());
    assert_eq!(regrowth.range(), Some(16..48));
    apply(&regrowth, &mirror, &mut gpu);
    assert_eq!(gpu, mirror);
    mirror.clear();
    let mut empty = UploadSpan::default();
    empty.force_full(mirror.len());
    assert_eq!(empty.range(), None);
}

#[test]
fn aligned_edge_stride_padding_is_preserved_between_changed_records() {
    const STRIDE: usize = 256;
    const RECORD: usize = 96;
    let mut mirror = vec![0; STRIDE * 3];
    let mut gpu = mirror.clone();
    let mut span = UploadSpan::default();
    for (slot, value) in [(0, 1), (2, 3)] {
        let start = slot * STRIDE;
        span.replace_bytes(start, &mut mirror[start..start + RECORD], &[value; RECORD]);
    }
    assert_eq!(span.range(), Some(0..STRIDE * 2 + RECORD));
    apply(&span, &mirror, &mut gpu);
    assert_eq!(gpu, mirror);
    for slot in 0..3 {
        assert!(
            gpu[slot * STRIDE + RECORD..(slot + 1) * STRIDE]
                .iter()
                .all(|byte| *byte == 0)
        );
    }
}

#[test]
fn replacing_gpu_or_staging_forces_the_whole_active_prefix() {
    let mut span = UploadSpan::default();
    span.mark(16..32);
    span.force_full(64);
    assert_eq!(span.range(), Some(0..64));
    assert_eq!(span.byte_count(), 64);
    let mirror = [11u8; 64];
    let mut new_gpu = [0; 64];
    apply(&span, &mirror, &mut new_gpu);
    assert_eq!(new_gpu, mirror);
    let mut new_staging = [0; 64];
    let mut span = UploadSpan::default();
    assert!(!span.replace_bytes(0, &mut new_staging, &[0; 64]));
    span.force_full(new_staging.len());
    apply(&span, &new_staging, &mut new_gpu);
    assert_eq!(new_gpu, new_staging);
    span.force_full(0);
    assert_eq!(span.byte_count(), 0);
}

#[test]
fn record_comparison_and_span_merging_allocate_nothing() {
    let mut records = [[0u32; 4]; 8];
    let ((span, records), allocations) = crate::test_allocations::count(|| {
        let mut span = UploadSpan::default();
        span.replace(16, &mut records[1], [1; 4]);
        span.replace(96, &mut records[6], [2; 4]);
        (span, records)
    });
    assert_eq!(allocations, 0);
    assert_eq!(span.range(), Some(16..112));
    assert_eq!(records[1], [1; 4]);
    assert_eq!(records[6], [2; 4]);
}
