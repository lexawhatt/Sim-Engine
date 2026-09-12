//! CPU-only public-API shaping/allocations; no window, device or rendering.

use sim_engine::{
    FontBudget, FontFace, LogicalPixels, PhysicalPerLogical, TextDirection, TextLayoutBudget,
    TextShapingSession, TextStyle,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn record_allocation() {
    let _ = COUNTING.try_with(|enabled| {
        if enabled.get() {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        // Forward the caller's allocator contract unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn measured(mut work: impl FnMut() -> Result<()>) -> Result<(usize, Duration)> {
    struct Stop;
    impl Drop for Stop {
        fn drop(&mut self) {
            COUNTING.set(false);
        }
    }
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    let stop = Stop;
    let started = Instant::now();
    work()?;
    let elapsed = started.elapsed();
    drop(stop);
    Ok((ALLOCATIONS.get(), elapsed))
}

fn main() -> Result<()> {
    let mut iterations = 1000;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--iterations" => {
                iterations = arguments.next().ok_or("missing iteration count")?.parse()?
            }
            "--help" | "-h" => {
                println!(
                    "CPU-only text shaping allocation benchmark: --iterations 1..100000 (default 1000)"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    if !(1..=100_000).contains(&iterations) {
        return Err("iterations must be in 1..=100000".into());
    }
    let font = FontFace::from_bytes(
        include_bytes!("assets/fonts/DejaVuSans.ttf").to_vec(),
        FontBudget::default(),
    )?;
    let budget = TextLayoutBudget::new(256, 256, 4096, 4096);
    println!(
        "CPU text only; per-thread allocation/reallocation calls include dependencies, exclude font load/session initialization/warmup/printing."
    );
    println!(
        "Scratch/line bytes are engine-owned capacity, not total process memory; parsed-face/plan/buffer capacities are not exposed by rustybuzz. Timings are diagnostic, not a release threshold."
    );
    for (name, strings, direction) in [
        (
            "latin32",
            [
                "0123456789abcdefghijklmnopqrstuv",
                "ABCDEFGHIJKLMNOPQRSTUV0123456789",
            ],
            TextDirection::Ltr,
        ),
        (
            "cyrillic_changing_length",
            ["Привет, мир!", "Состояние: готово"],
            TextDirection::Auto,
        ),
        (
            "combining",
            ["Cafe\u{301} q\u{301}", "A\u{30a} q\u{301} e\u{301}"],
            TextDirection::Auto,
        ),
        (
            "ligatures_changing_length",
            ["office ffi AV", "affinity AV ffi office"],
            TextDirection::Ltr,
        ),
        (
            "arabic_rtl",
            ["مرحبا بالعالم", "العالم"],
            TextDirection::Rtl,
        ),
        (
            "inferred_script_changes",
            ["Ready", "Готово"],
            TextDirection::Auto,
        ),
    ] {
        let style = TextStyle::new(LogicalPixels::new(24.0)?, PhysicalPerLogical::new(1.25)?)?
            .with_direction(direction);
        let mut session = TextShapingSession::new(&font, style, budget)?;
        let mut line = session.shape_line(strings[0])?;
        for index in 0..16 {
            session.update_line(&mut line, strings[index % 2])?;
            black_box(font.shape_line(strings[index % 2], &style, &budget)?);
        }
        let (fresh_calls, fresh_elapsed) = measured(|| {
            for index in 0..iterations {
                black_box(font.shape_line(strings[index % 2], &style, &budget)?);
            }
            Ok(())
        })?;
        let (session_calls, session_elapsed) = measured(|| {
            for index in 0..iterations {
                black_box(session.update_line(&mut line, strings[index % 2])?);
            }
            Ok(())
        })?;
        let same = strings[(iterations - 1) % 2];
        let (unchanged_calls, _) = measured(|| {
            for _ in 0..iterations {
                assert!(!session.update_line(&mut line, same)?);
            }
            Ok(())
        })?;
        println!(
            "{name}: changes={iterations}, fresh_allocations={fresh_calls}, session_allocations={session_calls}, unchanged_allocations={unchanged_calls}, fresh_cpu_us={:.3}, session_cpu_us={:.3}, scratch_bytes={}, line_bytes={}, cached_plans={}",
            fresh_elapsed.as_secs_f64() * 1e6 / iterations as f64,
            session_elapsed.as_secs_f64() * 1e6 / iterations as f64,
            session.allocation_bytes(),
            line.allocation_bytes(),
            session.cached_plan_count()
        );
    }
    Ok(())
}
