//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    })
    .contains(&n) || requested(n)
}

/// An optional one-shot request lets a tester capture after reaching the
/// scene, whose frame number depends on loading time. Create the file
/// named by `VICTORY_CAPTURE_REQUEST_FILE` to capture the next frame.
/// All draws and the present must agree: consume at most once per frame
/// and keep its result until the frame changes.
fn requested(n: u64) -> bool {
    static PATH: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    static FRAMES: std::sync::Mutex<std::collections::BTreeMap<u64, bool>> =
        std::sync::Mutex::new(std::collections::BTreeMap::new());
    let Some(path) = PATH.get_or_init(|| {
        std::env::var_os("VICTORY_CAPTURE_REQUEST_FILE").map(PathBuf::from)
    }) else {
        return false;
    };
    let mut frames = FRAMES.lock().unwrap_or_else(|e| e.into_inner());
    poll_request(&mut frames, n, path)
}

fn poll_request(
    frames: &mut std::collections::BTreeMap<u64, bool>,
    n: u64,
    path: &std::path::Path,
) -> bool {
    if let Some(&capture) = frames.get(&n) {
        return capture;
    }
    let capture = std::fs::remove_file(path).is_ok();
    frames.insert(n, capture);
    // Recording the next frame and presenting the preceding one can
    // interleave. Keep their decisions, with a small bounded history.
    while frames.len() > 8 {
        frames.pop_first();
    }
    if capture {
        tracing::info!(frame = n, "one-shot frame capture requested");
    }
    capture
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    let path = folder.join(format!("{name}.png"));
    match write_png(&path, width, height, pitch, bgra, rows) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

fn write_png(path: &std::path::Path, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) -> std::io::Result<()> {
    let (w, h) = ((width / 2).max(1), (height / 2).max(1));
    // Each row starts with a zero filter byte, followed by downsampled RGB.
    let row_bytes = w as usize * 3 + 1;
    let mut raw = vec![0; row_bytes * h as usize];
    for (y, scanline) in raw.chunks_exact_mut(row_bytes).enumerate() {
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for (x, pixel) in scanline[1..].chunks_exact_mut(3).enumerate() {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                pixel.copy_from_slice(&[p[2], p[1], p[0]]);
            } else {
                pixel.copy_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png)
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

const CRC32_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut byte = 0;
    while byte < table.len() {
        let mut crc = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
            bit += 1;
        }
        table[byte] = crc;
        byte += 1;
    }
    table
};

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc = (crc >> 8) ^ CRC32_TABLE[usize::from((crc as u8) ^ byte)];
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    if total == u64::MAX {
        return None;
    }
    // Our CPU readers take a slice of complete pitched rows, including
    // padding in the last row that the GPU copy does not require.
    let f = footprint.Footprint;
    let pitched_bytes = u64::from(f.RowPitch) * u64::from(f.Height);
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total.max(pitched_bytes),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    save_color_readback(&format!("frame-{frame}"), buffer, footprint);
}

/// Saves the preview actually copied into the swap chain's back buffer.
pub(crate) fn save_presented_preview(
    frame: u64,
    color: bool,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let mode = if color { "color" } else { "depth" };
    save_color_readback(&format!("frame-{frame}-presented-{mode}"), buffer, footprint);
}

fn save_color_readback(
    name: &str,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = f.RowPitch as usize * f.Height as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save_as(name, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

/// With `VICTORY_PRESENT_1440P` set: the latest `2560x1440` color render target
/// (`f28`/`f29`/`f87`/`f91`) written during the current frame, substituted as
/// plane 0 at present time when the title's `3840x2160` upscale pass outputs zero.
static LAST_1440P: std::sync::Mutex<Option<std::sync::Arc<crate::resource::HostTexture>>> =
    std::sync::Mutex::new(None);

fn present_1440p_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PRESENT_1440P").is_some())
}

/// Takes the latest `2560x1440` color render target noted this frame, if
/// `VICTORY_PRESENT_1440P` is enabled.
pub(crate) fn take_1440p_target() -> Option<std::sync::Arc<crate::resource::HostTexture>> {
    if !present_1440p_on() {
        return None;
    }
    LAST_1440P.lock().unwrap_or_else(|e| e.into_inner()).take()
}

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && std::env::var_os("VICTORY_DUMP_TARGETS").is_some())
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() {
        return;
    }
    if present_1440p_on()
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        // SAFETY: a live host texture.
        let d = unsafe { copy.texture.GetDesc() };
        if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
            && d.SampleDesc.Count == 1
            && d.Width == 2560
            && d.Height == 1440
            && matches!(d.Format.0, 28 | 29 | 87 | 91)
        {
            *LAST_1440P.lock().unwrap_or_else(|e| e.into_inner()) = Some(copy);
        }
    }
    crate::scene_color::note_bound(resource);
    if !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        if let Some((width, height, rgba)) = dump_target_rgba(host, texture, state) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:#x}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0,
            );
            save_as(&name, width, height, width as usize * 4, false, &rgba);
        }
    }
}

/// The latest usable depth sample, converted to tightly packed RGBA8.
pub(crate) struct ScenePreview {
    pub(crate) frame: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) rgba: Vec<u8>,
    pub(crate) color: bool,
    pub(crate) source_resource: Option<usize>,
    pub(crate) source_format: Option<u32>,
    pub(crate) depth_draws: Option<u32>,
}

const DEPTH_SAMPLE_GAP: u64 = 4;
const MAX_DEPTH_CANDIDATES: usize = 256;
static NEXT_DEPTH_SAMPLE: AtomicU64 = AtomicU64::new(32);
static SCENE_DEPTH: Mutex<SceneDepthCapture> = Mutex::new(SceneDepthCapture {
    presented: 0,
    drawn: 0,
    dispatched: 0,
    pending: BTreeMap::new(),
    latest: None,
});

struct SceneDepthCapture {
    presented: u64,
    drawn: u64,
    dispatched: u64,
    pending: BTreeMap<u64, DepthCandidates>,
    latest: Option<Arc<ScenePreview>>,
}

#[derive(Default)]
struct DepthCandidates {
    seen: HashSet<usize>,
    largest: Option<(u64, Arc<crate::resource::HostTexture>)>,
}

fn scene_depth_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("VICTORY_PREVIEW_SCENE_DEPTH").as_deref(), Ok("1")))
}

pub(crate) fn scene_depth_preview() -> Option<Arc<ScenePreview>> {
    if !scene_depth_on() {
        return None;
    }
    SCENE_DEPTH
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .latest
        .clone()
}

/// Keep the largest eligible depth target bound for the frame being built.
pub(crate) fn note_scene_depth(resource: crate::object::This) {
    if !scene_depth_on() {
        return;
    }
    let frame = crate::queue::presented() + 1;
    if frame < NEXT_DEPTH_SAMPLE.load(Relaxed) {
        return;
    }
    let mut capture = SCENE_DEPTH.lock().unwrap_or_else(|e| e.into_inner());
    // A recording thread may already be building the next frame.
    if frame <= capture.presented
        || frame > capture.presented.saturating_add(2)
        || frame < NEXT_DEPTH_SAMPLE.load(Relaxed)
    {
        return;
    }
    let candidates = capture.pending.entry(frame).or_default();
    if candidates.seen.len() >= MAX_DEPTH_CANDIDATES || !candidates.seen.insert(resource as usize) {
        return;
    }
    let Some(copy) = crate::resource::host_copy(resource) else { return };
    // SAFETY: the host texture is kept alive by the Arc.
    let desc = unsafe { copy.texture.GetDesc() };
    let (width, height) = (desc.Width, u64::from(desc.Height));
    if desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.SampleDesc.Count != 1
        || !desc.Flags.contains(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
        || !matches!(desc.Format.0, 19..=22 | 39..=47)
        || !(640..=4096).contains(&width)
        || !(360..=2304).contains(&height)
        || !(14 * height..=24 * height).contains(&(10 * width))
    {
        return;
    }
    let area = width * height;
    if candidates.largest.as_ref().is_none_or(|(largest, _)| area > *largest) {
        candidates.largest = Some((area, copy));
    }
}

/// At PresentX, read the submitted depth if host rendering reached a scene.
pub(crate) fn sample_scene_depth(frame: u64) {
    if !scene_depth_on() {
        return;
    }
    let texture = {
        let mut capture = SCENE_DEPTH.lock().unwrap_or_else(|e| e.into_inner());
        if frame <= capture.presented {
            return;
        }
        capture.presented = frame;
        let drawn = crate::draw::drawn();
        let dispatched = crate::draw::dispatched();
        let draws = drawn.saturating_sub(std::mem::replace(&mut capture.drawn, drawn));
        let dispatches =
            dispatched.saturating_sub(std::mem::replace(&mut capture.dispatched, dispatched));
        let candidates = capture.pending.remove(&frame);
        capture.pending.retain(|&pending, _| pending > frame);
        if frame < NEXT_DEPTH_SAMPLE.load(Relaxed)
            || draws < 64
            || (draws < 128 && dispatches < 16)
        {
            return;
        }
        let Some((_, texture)) = candidates.and_then(|candidates| candidates.largest) else {
            return;
        };
        let next = frame.saturating_add(DEPTH_SAMPLE_GAP);
        NEXT_DEPTH_SAMPLE.store(next, Relaxed);
        capture.pending.retain(|&pending, _| pending >= next);
        texture
    };
    let Some(host) = crate::host::host() else { return };
    let Ok(list) = host.new_list() else { return };
    // This is the only use in this list. Its prelude moves the depth
    // target to COPY_SOURCE under the host submission lock. Later title
    // lists transition it back to their required state at submission.
    let _ = list.track_texture_state(texture.clone(), D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    let Some(readback) = read_back(&host.device, &list.list, &texture.texture) else {
        return;
    };
    let Some((width, height, rgba)) = read_target_rgba(host, &list, readback, None) else {
        return;
    };
    if usable_depth(&rgba) {
        let mut capture = SCENE_DEPTH.lock().unwrap_or_else(|e| e.into_inner());
        if capture.latest.as_ref().is_none_or(|last| last.frame < frame) {
            capture.latest = Some(Arc::new(ScenePreview {
                frame,
                width,
                height,
                rgba,
                color: false,
                source_resource: None,
                source_format: None,
                depth_draws: None,
            }));
            tracing::info!(frame, width, height, "live scene depth preview updated");
        }
    }
}

/// Keep the last usable sample when a target was cleared or holds sparse noise.
fn usable_depth(rgba: &[u8]) -> bool {
    let mut bins = [0usize; 16];
    for pixel in rgba.chunks_exact(4).step_by(31) {
        bins[usize::from(pixel[0] >> 4)] += 1;
    }
    let count: usize = bins.iter().sum();
    let peak = bins.into_iter().max().unwrap_or(0);
    count - peak >= count.div_ceil(512).max(16)
}

fn dump_target_rgba(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
) -> Option<(u32, u32, Vec<u8>)> {
    let list = host.new_list().ok()?;
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        read
    };
    read_target_rgba(host, &list, read?, None)
}

/// Read a scene texture using the command list's submission-time state tracking.
pub(crate) fn read_scene_color(
    host: &crate::host::Host,
    copy: Arc<crate::resource::HostTexture>,
    format: DXGI_FORMAT,
) -> Option<(u32, u32, Vec<u8>)> {
    let list = host.new_list().ok()?;
    let readback = read_back(&host.device, &list.list, &copy.texture)?;
    let _ = list.track_texture_state(copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    read_target_rgba(host, &list, readback, Some(format))
}

fn read_target_rgba(
    host: &crate::host::Host,
    list: &crate::host::HostList,
    (buffer, footprint): (ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT),
    format: Option<DXGI_FORMAT>,
) -> Option<(u32, u32, Vec<u8>)> {
    list.close();
    host.execute(&[list]);
    if !host.capture_completed() {
        tracing::warn!("target readback fence did not complete");
        return None;
    }
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished and read_back allocated the full pitched rows.
    let rgba = unsafe {
        buffer.Map(0, None, Some(&mut data)).ok()?;
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        let format = format.unwrap_or(f.Format);
        let rgba = to_rgba(format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows);
        buffer.Unmap(0, None);
        rgba
    };
    if rgba.is_none() {
        tracing::info!(format = f.Format.0, "target not dumped: format");
    }
    rgba.map(|rgba| (f.Width, f.Height, rgba))
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    tracing::info!(format, low, high, "target values");
    for v in values {
        let c = v.map(|c| (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8);
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    probe_grid(device, list, texture, d)
}

fn probe_grid(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    d: D3D12_RESOURCE_DESC,
) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Record a sparse diagnostic grid on a scene candidate already checked for
/// dimensions, sample count, and format by the preview.
pub(crate) fn record_scene_grid(
    host: &crate::host::Host,
    list: &crate::host::HostList,
    copy: Arc<crate::resource::HostTexture>,
    desc: D3D12_RESOURCE_DESC,
) -> Option<ID3D12Resource> {
    let buffer = probe_grid(&host.device, &list.list, &copy.texture, desc)?;
    let _ = list.track_texture_state(copy, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
    Some(buffer)
}

/// Whether the completed grid contains a finite RGB component above the floor.
pub(crate) fn scene_grid_has_color(buffer: &ID3D12Resource, format: DXGI_FORMAT) -> bool {
    let floor = if matches!(format.0, 10 | 26) { 0.0001 } else { 8.0 / 255.0 };
    let mut mapped = std::ptr::null_mut();
    // SAFETY: the caller checked the completion fence and the buffer covers all grid slots.
    unsafe {
        if buffer.Map(0, None, Some(&mut mapped)).is_err() {
            return false;
        }
        let bytes = std::slice::from_raw_parts(
            mapped.cast::<u8>(),
            (SLOT * u64::from(GRID * GRID)) as usize,
        );
        let found = bytes.chunks_exact(SLOT as usize).any(|pixel| {
            let packed = u32::from_le_bytes(pixel[..4].try_into().expect("grid pixel"));
            let rgb = match format.0 {
                10 => [0, 2, 4].map(|i| half(u16::from_le_bytes([pixel[i], pixel[i + 1]]))),
                24 => [0, 10, 20].map(|i| ((packed >> i) & 0x3ff) as f32 / 1023.0),
                26 => [
                    small_float(packed & 0x7ff, 6),
                    small_float((packed >> 11) & 0x7ff, 6),
                    small_float(packed >> 22, 5),
                ],
                // For this threshold the RGB and BGR channel orders are equivalent.
                _ => [0, 1, 2].map(|i| f32::from(pixel[i]) / 255.0),
            };
            rgb.into_iter().any(|channel| channel.is_finite() && channel.abs() > floor)
        });
        buffer.Unmap(0, None);
        found
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }
    #[test]
    fn a_capture_request_survives_interleaved_recording_and_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("capture.request");
        let mut frames = std::collections::BTreeMap::new();
        assert!(!super::poll_request(&mut frames, 10, &path));
        std::fs::write(&path, []).unwrap();
        // A request arriving mid-frame waits for the next complete frame.
        assert!(!super::poll_request(&mut frames, 10, &path));
        assert!(path.exists());
        assert!(super::poll_request(&mut frames, 11, &path));
        assert!(!path.exists());
        // Presenting the preceding frame cannot erase the next decision.
        assert!(!super::poll_request(&mut frames, 10, &path));
        assert!(super::poll_request(&mut frames, 11, &path));
        assert!(!super::poll_request(&mut frames, 12, &path));
        std::fs::write(&path, []).unwrap();
        assert!(super::poll_request(&mut frames, 11, &path));
        assert!(path.exists());
        assert!(super::poll_request(&mut frames, 13, &path));
        assert!(!path.exists());
    }

}


/// Saves each newly selected preview after its present succeeds. The ring holds
/// 128 source PNGs and matching metadata, independently of periodic final captures.
/// These are normalized source pixels before swap-chain resizing/compositing.
pub(crate) fn archive_preview(present_frame: u64, image: &ScenePreview) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static NEXT: AtomicU64 = AtomicU64::new(0);
    if !*ENABLED.get_or_init(|| std::env::var("VICTORY_CAPTURE_PREVIEW").is_ok_and(|v| v == "1")) {
        return;
    }
    let Some(folder) = folder() else { return };
    let directory = folder.join("preview-ring");
    let sequence = NEXT.fetch_add(1, Relaxed);
    let slot = sequence % 128;
    let result = write_preview_archive(&directory, sequence, present_frame, image);
    match result {
        Ok(()) => tracing::info!(sequence, slot, present_frame, source_frame = image.frame, "preview archived"),
        Err(error) => tracing::warn!(%error, sequence, slot, "preview archive failed"),
    }
}

fn write_preview_archive(directory: &std::path::Path, sequence: u64, present_frame: u64, image: &ScenePreview) -> std::io::Result<()> {
    let slot = sequence % 128;
    std::fs::create_dir_all(&directory)?;
    let stem = format!("slot-{slot:03}");
    write_png(&directory.join(format!("{stem}.png")), image.width, image.height,
        image.width as usize * 4, false, &image.rgba)?;
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default().as_millis();
    let kind = if image.color { "color" } else { "depth" };
    let resource = image.source_resource.map(|v| format!("\"{v:#x}\""))
        .unwrap_or_else(|| "null".into());
    let format = image.source_format.map(|v| v.to_string()).unwrap_or_else(|| "null".into());
    let draws = image.depth_draws.map(|v| v.to_string()).unwrap_or_else(|| "null".into());
    let metadata = format!("{{\"sequence\":{sequence},\"slot\":{slot},\"present_frame\":{present_frame},\"source_frame\":{},\"unix_ms\":{now},\"kind\":\"{kind}\",\"resource\":{resource},\"format\":{format},\"depth_draws\":{draws},\"width\":{},\"height\":{},\"pixels\":\"normalized source, half-size RGB PNG\",\"present_result\":\"S_OK\"}}\n", image.frame, image.width, image.height);
    std::fs::write(directory.join(format!("{stem}.json")), metadata)
}

#[cfg(test)]
mod preview_archive_tests {
    #[test]
    fn archive_wraps_without_mixing_source_metadata_and_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let mut image = super::ScenePreview {
            frame: 0, width: 4, height: 4, rgba: vec![127; 4 * 4 * 4], color: true,
            source_resource: Some(0x1234), source_format: Some(28), depth_draws: Some(64),
        };
        for sequence in 0..129 {
            image.frame = sequence * 10;
            super::write_preview_archive(directory.path(), sequence, sequence * 10 + 1, &image).unwrap();
        }
        let files: Vec<_> = std::fs::read_dir(directory.path()).unwrap().collect();
        assert_eq!(files.len(), 256);
        let png = std::fs::read(directory.path().join("slot-000.png")).unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        assert_eq!(u32::from_be_bytes(png[16..20].try_into().unwrap()), 2);
        let metadata = std::fs::read_to_string(directory.path().join("slot-000.json")).unwrap();
        assert!(metadata.contains("\"sequence\":128"));
        assert!(metadata.contains("\"source_frame\":1280"));
        assert!(metadata.contains("\"present_frame\":1281"));
        assert!(metadata.contains("\"resource\":\"0x1234\""));
    }
}

