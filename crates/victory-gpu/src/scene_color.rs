//! Optional in-window preview of game-produced color before composition.

use std::collections::{BTreeMap, HashMap};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering::Relaxed},
};

use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::DXGI_FORMAT};

use crate::{capture, host::Host, object::This, resource::HostTexture};

const MAX_TRACKED: usize = 512;
const PROBE_COUNT: usize = 8;
const SAMPLE_GAP: u64 = 3;
const HOLD_FRAMES: u64 = 10;
static NEXT: AtomicU64 = AtomicU64::new(32);
static STATE: Mutex<State> = Mutex::new(State {
    presented: 0,
    drawn: 0,
    sample: 0,
    pending: BTreeMap::new(),
    latest: None,
});

struct State {
    presented: u64,
    drawn: u64,
    sample: usize,
    pending: BTreeMap<u64, Frame>,
    latest: Option<Arc<capture::ScenePreview>>,
}

#[derive(Default)]
struct Frame {
    // A rejected resource stays in the map to avoid recurring descriptor queries.
    targets: HashMap<usize, Option<Target>>,
    depths: Vec<(u64, u32)>,
    depth_dimensions: HashMap<usize, Option<(u64, u32)>>,
}

struct Target {
    copy: Arc<HostTexture>,
    desc: D3D12_RESOURCE_DESC,
    format: DXGI_FORMAT,
    depth_draws: u32,
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("VICTORY_PREVIEW_SCENE_COLOR").as_deref(),
            Ok("1")
        )
    })
}

fn with_pending(f: impl FnOnce(&mut Frame)) {
    if !enabled() {
        return;
    }
    let frame = crate::queue::presented() + 1;
    if frame < NEXT.load(Relaxed) {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if frame > state.presented
        && frame <= state.presented.saturating_add(2)
        && frame >= NEXT.load(Relaxed)
    {
        f(state.pending.entry(frame).or_default());
    }
}

fn dimensions(desc: &D3D12_RESOURCE_DESC) -> bool {
    let h = u64::from(desc.Height);
    desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.SampleDesc.Count == 1
        && desc.DepthOrArraySize == 1
        && (640..=4096).contains(&desc.Width)
        && (360..=2304).contains(&h)
        && (14 * h..=24 * h).contains(&(10 * desc.Width))
}

fn format(raw: DXGI_FORMAT, view: Option<DXGI_FORMAT>) -> Option<DXGI_FORMAT> {
    let typed = view.unwrap_or(raw);
    matches!(
        (raw.0, typed.0),
        (9..=10, 10) | (23..=24, 24) | (26, 26) | (27..=29, 28..=29) | (87 | 90 | 91, 87 | 91)
    )
    .then_some(typed)
}

impl Frame {
    fn record(&mut self, guest: usize, paired: bool, view: Option<DXGI_FORMAT>) {
        if let Some(Some(target)) = self.targets.get_mut(&guest) {
            if paired {
                target.depth_draws = target.depth_draws.saturating_add(1);
            }
            return;
        }
        // A real RTV view can resolve an earlier rejected typeless binding.
        if (self.targets.contains_key(&guest) && view.is_none())
            || self.targets.len() >= MAX_TRACKED
        {
            return;
        }
        let target = crate::resource::host_copy(guest as This).and_then(|copy| {
            // SAFETY: the Arc keeps the texture live.
            let desc = unsafe { copy.texture.GetDesc() };
            if !dimensions(&desc)
                || desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
                || desc.Flags.0
                    & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
                        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0)
                    == 0
            {
                return None;
            }
            let format = format(desc.Format, view)?;
            Some(Target {
                copy,
                desc,
                format,
                depth_draws: u32::from(paired),
            })
        });
        self.targets.insert(guest, target);
    }
}

/// RTV and UAV bindings provide secondary candidates for post-lighting passes.
pub(crate) fn note_bound(resource: This) {
    with_pending(|frame| frame.record(resource as usize, false, None));
}

/// Successful graphics commands with scene-sized depth give stronger candidates.
pub(crate) fn note_draw(
    depth: usize,
    targets: &[usize],
    views: &[Option<D3D12_RENDER_TARGET_VIEW_DESC>],
) {
    if depth == 0 || targets.is_empty() {
        return;
    }
    with_pending(|frame| {
        let Some(dims) = *frame.depth_dimensions.entry(depth).or_insert_with(|| {
            let copy = crate::resource::host_copy(depth as This)?;
            // SAFETY: the Arc keeps the depth texture live.
            let desc = unsafe { copy.texture.GetDesc() };
            dimensions(&desc).then_some((desc.Width, desc.Height))
        }) else {
            return;
        };
        if !frame.depths.contains(&dims) {
            frame.depths.push(dims);
        }
        for (&target, view) in targets.iter().zip(views) {
            let format = match view {
                Some(d) if d.ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D => {
                    // SAFETY: the two-dimensional view tag selects this union member.
                    let subresource = unsafe { d.Anonymous.Texture2D };
                    if subresource.MipSlice != 0 || subresource.PlaneSlice != 0 {
                        continue;
                    }
                    Some(d.Format)
                }
                Some(_) => continue,
                None => None,
            };
            frame.record(target, true, format);
        }
    });
}

/// Wait for depth-tested scene work before selecting any color surface.
pub(crate) fn sample(host: &Host, frame: u64, presented_resource: usize) {
    if !enabled() {
        return;
    }
    let (mut targets, round) = {
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if frame <= state.presented {
            return;
        }
        state.presented = frame;
        let drawn = crate::draw::drawn();
        let draws = drawn.saturating_sub(std::mem::replace(&mut state.drawn, drawn));
        let candidates = state.pending.remove(&frame);
        state.pending.retain(|&n, _| n > frame);
        let Some(candidates) = candidates else { return };
        if draws < 64 || candidates.depths.is_empty() || frame < NEXT.load(Relaxed) {
            return;
        }
        let targets: Vec<_> = candidates
            .targets
            .into_iter()
            .filter_map(|(guest, target)| {
                let target = target?;
                (guest != presented_resource
                    && candidates
                        .depths
                        .contains(&(target.desc.Width, target.desc.Height)))
                .then_some((guest, target))
            })
            .collect();
        if targets.is_empty() {
            return;
        }
        let round = state.sample;
        state.sample = state.sample.wrapping_add(1);
        NEXT.store(frame.saturating_add(SAMPLE_GAP), Relaxed);
        (targets, round)
    };
    targets.sort_unstable_by_key(|(guest, target)| {
        (
            std::cmp::Reverse(target.depth_draws > 0),
            std::cmp::Reverse(matches!(target.format.0, 28 | 29 | 87 | 91)),
            std::cmp::Reverse(target.depth_draws),
            *guest,
        )
    });
    let count = targets.len();
    // Rotate across targets even when the same high-ranked buffers stay empty.
    let start = round.wrapping_mul(PROBE_COUNT) % count;
    let shortlist: Vec<_> = (0..count.min(PROBE_COUNT))
        .map(|i| (start + i) % count)
        .collect();
    let mut positive = probe(host, &targets, &shortlist);
    positive.truncate(2);
    // A sparse colored fragment can miss every point of the grid.
    let blind = (start + round) % count;
    if !positive.contains(&blind) {
        positive.push(blind);
    }
    tracing::info!(
        frame,
        candidates = count,
        probed = shortlist.len(),
        readable = positive.len(),
        "scene color probe"
    );
    for index in positive.into_iter().take(3) {
        let (guest, target) = &targets[index];
        let Some((width, height, rgba)) = capture::read_scene_color(host, target.copy.clone(), target.format) else {
            continue;
        };
        if !scene_pixels(&rgba) {
            continue;
        }
        let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if state
            .latest
            .as_ref()
            .is_none_or(|image| image.frame < frame)
        {
            state.latest = Some(Arc::new(capture::ScenePreview {
                frame,
                width,
                height,
                rgba,
                color: true,
            }));
            tracing::info!(
                frame,
                resource = format_args!("{guest:#x}"),
                width,
                height,
                format = target.format.0,
                depth_draws = target.depth_draws,
                "live scene color preview updated"
            );
        }
        break;
    }
}

fn probe(host: &Host, targets: &[(usize, Target)], indices: &[usize]) -> Vec<usize> {
    let Ok(list) = host.new_list() else {
        return Vec::new();
    };
    let buffers: Vec<_> = indices
        .iter()
        .copied()
        .filter_map(|i| {
            let target = &targets[i].1;
            capture::record_scene_grid(host, &list, target.copy.clone(), target.desc)
                .map(|buffer| (i, buffer))
        })
        .collect();
    if buffers.is_empty() {
        return Vec::new();
    }
    list.close();
    host.execute(&[&list]);
    if !host.capture_completed() {
        return Vec::new();
    }
    buffers
        .into_iter()
        .filter_map(|(i, buffer)| {
            capture::scene_grid_has_color(&buffer, targets[i].1.format).then_some(i)
        })
        .collect()
}

/// Accept stored RGB variation. Reject clears, grey buffers and tiny noise.
fn scene_pixels(rgba: &[u8]) -> bool {
    let mut bands = [false; 4096];
    let mut distinct = 0;
    let mut colored = 0;
    for p in rgba.chunks_exact(4).step_by(17) {
        let (r, g, b) = (p[0], p[1], p[2]);
        let high = r.max(g).max(b);
        if high.saturating_sub(r.min(g).min(b)) >= 8 && high >= 12 {
            colored += 1;
            let band =
                (usize::from(r >> 4) << 8) | (usize::from(g >> 4) << 4) | usize::from(b >> 4);
            if !std::mem::replace(&mut bands[band], true) {
                distinct += 1;
            }
        }
    }
    colored >= 8 && distinct >= 2
}

pub(crate) fn preview(frame: u64) -> Option<Arc<capture::ScenePreview>> {
    if !enabled() {
        return None;
    }
    STATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .latest
        .as_ref()
        .filter(|image| frame.saturating_sub(image.frame) <= HOLD_FRAMES)
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::scene_pixels;

    #[test]
    fn needs_shape_and_chroma() {
        let mut pixels = vec![0u8; 1024 * 4];
        assert!(!scene_pixels(&pixels));
        for rgb in pixels.chunks_exact_mut(4) {
            rgb.copy_from_slice(&[42, 42, 42, 255]);
        }
        assert!(!scene_pixels(&pixels));
        for rgb in pixels.chunks_exact_mut(4) {
            rgb.copy_from_slice(&[96, 42, 42, 255]);
        }
        assert!(!scene_pixels(&pixels));
        for rgb in pixels.chunks_exact_mut(4).take(256) {
            rgb.copy_from_slice(&[140, 80, 42, 255]);
        }
        assert!(scene_pixels(&pixels));
    }
}
