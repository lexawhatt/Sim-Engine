#!/usr/bin/env bash
# No GPU/Cargo: simulate both mandatory-test outcomes and adapter evidence.
set -euo pipefail
[[ "$*" == "test --manifest-path $DENSE_SEMANTICS_TEST_SOURCE/Cargo.toml --offline --locked --all-features --lib renderer::tests::offscreen_gpu_readback_verifies_camera_depth_and_clip_contract -- --exact --nocapture --test-threads=1" ]]
[[ "$WGPU_BACKEND" == vulkan && "$SIM_ENGINE_REQUIRE_GPU_TESTS" == 1 && "$SIM_ENGINE_REQUIRE_VULKAN" == 1 ]]
[[ "$SIM_ENGINE_REQUIRE_ADAPTER_IDENTITY" == 0 && "$SIM_ENGINE_REQUIRE_PRODUCTION_SURFACE_FORMAT" == 1 ]]
[[ -z "${SIM_ENGINE_REQUIRED_ADAPTER_NAME+x}${SIM_ENGINE_REQUIRED_ADAPTER_VENDOR+x}${SIM_ENGINE_REQUIRED_ADAPTER_DEVICE+x}${SIM_ENGINE_REQUIRED_ADAPTER_BACKEND+x}" ]]
printf '%s %s\n' "$SIM_ENGINE_GPU_SURFACE_FORMAT" "$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT" >>"$DENSE_SEMANTICS_TEST_CALLS"
mode=${DENSE_SEMANTICS_TEST_MODE:-complete}
revision=$SIM_ENGINE_RELEASE_SHA
pci=$SIM_ENGINE_REQUIRED_ADAPTER_PCI_BUS_ID
format=$SIM_ENGINE_GPU_SURFACE_FORMAT
samples=$SIM_ENGINE_GPU_SURFACE_SAMPLE_COUNT
backend=Vulkan
case "$mode" in
    wrong_sha) revision=0000000000000000000000000000000000000000 ;;
    wrong_pci) pci=0000:02:00.0 ;;
    wrong_format) format=Rgba16Float ;;
    wrong_samples) samples=16 ;;
    wrong_backend) backend=Gl ;;
esac
printf 'vcs_sha=%s\nbackend=%s\npci_bus_id=%s\noracle_format=%s\noracle_sample_count=%s\n' \
    "$revision" "$backend" "$pci" "$format" "$samples" >"$SIM_ENGINE_GPU_EVIDENCE_PATH"
if [[ "$mode" == duplicate_key ]]; then printf 'backend=Gl\n' >>"$SIM_ENGINE_GPU_EVIDENCE_PATH"; fi
if [[ "$mode" == failure ]]; then echo 'test result: FAILED. 0 passed; 1 failed;'; exit 17; fi
if [[ "$mode" == zero_tests ]]; then
    printf 'running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 414 filtered out; finished in 0.00s\n'
elif [[ "$mode" == ignored ]]; then
    printf 'running 1 test\ntest result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 413 filtered out; finished in 0.00s\n'
elif [[ "$mode" != no_footer ]]; then
    printf 'running 1 test\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 413 filtered out; finished in 0.01s\n'
fi
