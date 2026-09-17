use super::*;
#[test]
fn region_source_stride_is_checked_without_allocation() {
    let region = ImageTexelRect::new(1, 1, 2, 2).unwrap();
    assert_eq!(validate_layout(region, &[0; 20], 12), Ok((8, 16)));
    for (data, stride) in [
        (&[0; 19][..], 12),
        (&[0; 16][..], 7),
        (&[0; 16][..], usize::MAX),
    ] {
        assert!(validate_layout(region, data, stride).is_err());
    }
}
