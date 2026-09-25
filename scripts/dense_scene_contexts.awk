# Fail-closed parser for actual workload/recovery context, not an adapter probe.
# Rust Debug strings may contain spaces, escaped quotes, or text resembling keys.
function fail(message) {
    print FILENAME ":" FNR ": " message > "/dev/stderr"
    failed = 1
    exit 1
}

function fields(line, prefix, result,    rest,key,value,ch,quoted,escaped,depth,i) {
    delete result
    if (substr(line, 1, length(prefix) + 1) != prefix " ") fail("malformed " prefix)
    rest = substr(line, length(prefix) + 2)
    while (length(rest)) {
        if (!match(rest, /^[a-z_][a-z_0-9]*=/)) fail("malformed context field")
        key = substr(rest, 1, RLENGTH - 1)
        if (key in result) fail("duplicate field " key)
        rest = substr(rest, RLENGTH + 1)
        value = ""; quoted = 0; escaped = 0; depth = 0
        for (i = 1; i <= length(rest); i++) {
            ch = substr(rest, i, 1)
            if (escaped) escaped = 0
            else if (quoted && ch == "\\") escaped = 1
            else if (ch == "\"") quoted = !quoted
            else if (!quoted) {
                if (ch == "(" || ch == "[") depth++
                if (ch == ")" || ch == "]") depth--
                if (depth < 0) fail("unbalanced context field")
                if (ch == " " && depth == 0) break
            }
            value = value ch
        }
        if (!length(value) || quoted || escaped || depth) fail("incomplete context field " key)
        result[key] = value
        rest = substr(rest, i + 1)
    }
}

function finish_file() {
    if (previous_file == "") return
    if (contexts == 0) fail("missing workload context")
    if (restored != 1 || recovery_drawn != 1 || recovery_pending) fail("missing complete recovery context")
}

BEGIN {
    if (expected_sha == "" || expected_pci == "") fail("missing required revision/physical adapter")
    required = "mode labels constructions phase trial generation fixture_version glyph_batch_chunks vcs_sha adapter backend vendor device pci driver driver_info physical scale format samples present refresh_millihertz gpu_timing"
    required_count = split(required, keys, " ")
}

FNR == 1 {
    finish_file()
    previous_file = FILENAME
    contexts = 0; restored = 0; recovery_drawn = 0; recovery_pending = 0; recovery_contexts = 0
}

/^DENSE_CONTEXT/ {
    fields($0, "DENSE_CONTEXT", value)
    for (k = 1; k <= required_count; k++) if (!(keys[k] in value)) fail("missing field " keys[k])
    if (value["vcs_sha"] != "Some(\"" expected_sha "\")") fail("context revision mismatch")
    if (value["pci"] != "\"" expected_pci "\"") fail("context physical adapter mismatch")
    if (value["backend"] != "vulkan") fail("context backend is not Vulkan")
    if (value["format"] !~ /^(Rgba8UnormSrgb|Rgba8Unorm|Bgra8UnormSrgb|Bgra8Unorm)$/) fail("unsupported surface format")
    if (value["samples"] !~ /^(1|2|4|8|16)$/) fail("invalid MSAA sample count")
    if (value["gpu_timing"] != "Enabled") fail("context lacks required GPU timing")
    if (value["adapter"] !~ /^".*"$/ || value["driver"] !~ /^".*"$/ || value["driver_info"] !~ /^".*"$/) fail("malformed driver/adapter strings")
    numeric_count = split("constructions trial generation fixture_version glyph_batch_chunks vendor device", numeric, " ")
    for (k = 1; k <= numeric_count; k++) if (value[numeric[k]] !~ /^[0-9]+$/) fail("invalid numeric context field " numeric[k])
    if (value["phase"] !~ /^(idle|pan|zoom|drag)$/) fail("invalid workload phase")
    if (value["mode"] !~ /^(Streaming|Prepared|Analytic)$/ || value["labels"] !~ /^(Off|Runs|Batch)$/) fail("invalid workload mode")
    if (value["physical"] !~ /^\[[0-9]+, [0-9]+\]$/ || value["scale"] !~ /^[0-9]+([.][0-9]+)?$/) fail("invalid output dimensions/scale")
    if (value["present"] !~ /^(Immediate|Mailbox|Fifo|FifoRelaxed|AutoVsync|AutoNoVsync)$/) fail("invalid presentation mode")
    if (value["refresh_millihertz"] !~ /^(None|Some\([0-9]+\))$/) fail("invalid refresh metadata")
    contexts++
    if (recovery_pending) recovery_contexts++
    file = FILENAME
    sub(/^.*\//, "", file)
    print file "\t" FNR "\t" value["format"] "\t" value["samples"]
}

/^DENSE_RECOVERY restored_cpu_ms=/ {
    if (restored || contexts == 0) fail("unexpected restoration boundary")
    restored++
    recovery_pending = 1
}

/^DENSE_RECOVERY status=drawn / {
    if (!recovery_pending || recovery_contexts == 0 || recovery_drawn) fail("recovery Drawn lacks post-restoration context")
    recovery_pending = 0
    recovery_drawn++
}

/^DENSE_RESTART/ {
    if (recovery_pending) fail("output changed before recovery was confirmed")
}

END {
    if (!failed) {
        if (previous_file == "") fail("no workload logs supplied")
        finish_file()
    }
}
