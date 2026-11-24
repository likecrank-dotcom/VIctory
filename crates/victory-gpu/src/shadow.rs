//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, by title page, covering
    /// the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
}

/// A copy's bytes before a submission ran.
#[derive(Default)]
struct Before {
    /// By title page.
    pages: BTreeMap<usize, Snap>,
    /// The ranges taken (offsets in the copy, merged where they meet): a
    /// range declared again (the world's are, dozens of times) is inside
    /// one and needs no walk over its pages.
    spans: Vec<(usize, usize)>,
}

/// A title page's bytes in the copy before the submission ran: `[lo, hi)`
/// of `bytes` (offsets in the page; a page's buffer is a page long).
struct Snap {
    lo: usize,
    hi: usize,
    bytes: Box<[u8]>,
}

/// Page buffers kept for reuse (already touched, so no page faults). A
/// snapshot as one buffer grown to the submission's bounding range took a
/// fresh buffer whenever the few kept were in use or too small, and copied
/// all of the old one into it at each growth.
static PAGES: Mutex<Vec<Box<[u8]>>> = Mutex::new(Vec::new());
/// At most this many page buffers kept (256 MiB).
const KEPT: usize = 4096;

/// A page buffer, from the pool if it has one.
fn page_buffer() -> Box<[u8]> {
    let kept = PAGES.lock().unwrap_or_else(|e| e.into_inner()).pop();
    kept.unwrap_or_else(|| vec![0; PAGE].into_boxed_slice())
}

/// Returns page buffers to the pool.
fn give_back(buffers: impl IntoIterator<Item = Box<[u8]>>) {
    let mut pool = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    for b in buffers {
        if pool.len() >= KEPT {
            break;
        }
        pool.push(b);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per title page and submission, each page's part grown to cover every
    /// range the submission declares in it (the world declares overlapping
    /// views of its 174 MB heap dozens of times), parts newly covered taken
    /// from the copy as it is now, in reused page buffers.
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        // In-page offsets `[a, b)` of `page`, from the copy.
        // SAFETY: inside the mapping (the caller passes offsets in the buffer).
        let host = |page: usize, a: usize, b: usize| unsafe {
            std::slice::from_raw_parts(self.data.add(page * PAGE + a - start), b - a)
        };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut *before;
        if from >= to || before.spans.iter().any(|&(a, b)| a <= from && to <= b) {
            return;
        }
        // An earlier snapshot can span a gap that was not declared yet.
        // Capture its refresh now, rather than treat it as a later GPU write.
        let fresh = outside(from, to, before.spans.clone());
        let (mut a, mut b) = (from, to);
        before.spans.retain(|&(x, y)| {
            let meets = x <= b && a <= y;
            if meets {
                (a, b) = (a.min(x), b.max(y));
            }
            !meets
        });
        before.spans.push((a, b));
        let mut at = from;
        while at < to {
            let (page, lo) = ((start + at) / PAGE, (start + at) % PAGE);
            let hi = lo + (PAGE - lo).min(to - at);
            let snap = before.pages.entry(page).or_insert_with(|| Snap {
                lo,
                hi: lo,
                bytes: page_buffer(),
            });
            let previous = (snap.lo, snap.hi);
            if lo < snap.lo {
                snap.bytes[lo..snap.lo].copy_from_slice(host(page, lo, snap.lo));
                snap.lo = lo;
            }
            if hi > snap.hi {
                let from = snap.hi.max(snap.lo);
                snap.bytes[from..hi].copy_from_slice(host(page, from, hi));
                snap.hi = hi;
            }
            // Newly covered edges were copied above. Refresh only fresh
            // portions inside the old extent; previously declared bytes
            // must keep the original snapshot across overlapping ranges.
            let page_start = page * PAGE;
            for &(a, b) in &fresh {
                let a = (start + a).max(page_start + previous.0);
                let b = (start + b).min(page_start + previous.1);
                if a < b {
                    let (a, b) = (a - page_start, b - page_start);
                    snap.bytes[a..b].copy_from_slice(host(page, a, b));
                }
            }
            at += hi - lo;
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if from < to {
            // By title page: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            let mut scratch = page_buffer();
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let (page, lo) = ((start + at) / PAGE, (start + at) % PAGE);
                let n = (PAGE - lo).min(to - at);
                let Some(snap) = snapshot
                    .pages
                    .get(&page)
                    .filter(|s| s.lo <= lo && lo + n <= s.hi)
                else {
                    at += n;
                    continue;
                };
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &snap.bytes[lo..lo + n];
                let title = &mut scratch[..n];
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    if changed {
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            give_back([scratch]);
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.retain(|&r| r != (address, address + len));
        if pending.is_empty() {
            let taken = std::mem::take(&mut *snapshot);
            give_back(taken.pages.into_values().map(|s| s.bytes));
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

/// Portions not yet declared by this submission, in copy-relative offsets.
fn outside(from: usize, to: usize, mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let (mut at, mut parts) = (from, Vec::new());
    for (a, b) in ranges {
        if at < to && a > at {
            parts.push((at, a.min(to)));
        }
        at = at.max(b);
    }
    if at < to { parts.push((at, to)); }
    parts
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy of title memory `[at, at + len)` holding the same bytes, if
    /// there is a host device.
    fn shadow_of(at: usize, len: usize) -> Option<Shadow> {
        let (buffer, data) = host()?.shared_buffer((len as u64).next_multiple_of(PAGE as u64))?;
        // SAFETY: the mapping holds `len` bytes; the title memory is ours.
        unsafe { std::ptr::copy_nonoverlapping(at as *const u8, data, len) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        Some(Shadow::new(buffer, data, gpu, at as u64, len as u64))
    }

    /// Two overlapping ranges shaders write, declared with a refresh in
    /// between: only the bytes the GPU changed go back, the title's own
    /// writes meanwhile stay, and the page buffers return to the pool.
    #[test]
    fn copy_back_takes_only_what_the_gpu_wrote() {
        let mut memory = vec![1u8; 4 * PAGE];
        // Off a page boundary, over three title pages.
        let base = (memory.as_mut_ptr() as usize).next_multiple_of(PAGE) + 100;
        let len = 2 * PAGE + 500;
        let Some(shadow) = shadow_of(base, len) else {
            return;
        };
        // SAFETY: offsets inside the title range and the copy.
        let title = |at: usize| unsafe { (base as *mut u8).add(at) };
        // SAFETY: as above.
        let host = |at: usize| unsafe { shadow.data.add(at) };
        // SAFETY: (all writes below) bytes of the title range and the copy,
        // which nothing else uses.
        unsafe {
            shadow.writing(base + 10, PAGE + 10);
            // A refresh next to the first range, in its last page.
            title(PAGE + 100).write(7);
            host(PAGE + 100).write(7);
            shadow.writing(base + PAGE + 50, PAGE + 50);
            // The GPU writes in both ranges; the title writes meanwhile.
            host(20).write(0xaa);
            host(PAGE + 200).write(0xbb);
            title(30).write(0xcc);
        }
        assert!(shadow.copy_back(base + 10, PAGE + 10) > 0);
        assert!(shadow.copy_back(base + PAGE + 50, PAGE + 50) > 0);
        // SAFETY: as above.
        unsafe {
            assert_eq!(title(20).read(), 0xaa);
            assert_eq!(title(PAGE + 200).read(), 0xbb);
            assert_eq!(title(30).read(), 0xcc, "the title's own write stays");
            assert_eq!(host(30).read(), 0xcc, "and the copy takes it");
            assert_eq!(title(PAGE + 100).read(), 7);
            assert_eq!(title(40).read(), 1);
        }
        assert!(shadow.before.lock().unwrap().pages.is_empty());
        assert!(PAGES.lock().unwrap().len() >= 3);
        drop(memory);
    }

    /// A range declared again, or inside one declared, takes nothing new.
    #[test]
    fn declaring_a_range_again_keeps_its_snapshot() {
        let memory = vec![3u8; 3 * PAGE];
        let base = memory.as_ptr() as usize;
        let Some(shadow) = shadow_of(base, 3 * PAGE) else {
            return;
        };
        shadow.writing(base, 2 * PAGE);
        // SAFETY: a byte of the copy, which nothing else uses: a GPU write
        // between two dispatches declaring the range.
        unsafe { shadow.data.add(5).write(9) };
        shadow.writing(base, 2 * PAGE);
        shadow.writing(base + 4, 100);
        {
            let before = shadow.before.lock().unwrap();
            assert_eq!(before.spans, [(0, 2 * PAGE)]);
            let at = base + 5;
            assert_eq!(before.pages[&(at / PAGE)].bytes[at % PAGE], 3);
        }
        shadow.copy_back(base, 2 * PAGE);
        shadow.copy_back(base + 4, 100);
        // SAFETY: title memory of ours.
        assert_eq!(unsafe { (base as *const u8).add(5).read() }, 9);
        assert!(shadow.before.lock().unwrap().pages.is_empty());
        drop(memory);
    }

    /// Refreshing an undeclared gap inside one snapshot page must not be
    /// mistaken for a GPU write and overwrite a later CPU write.
    #[test]
    fn refreshed_gap_inside_a_page_keeps_the_titles_newer_byte() {
        let mut memory = vec![0u8; 3 * PAGE];
        let base = (memory.as_mut_ptr() as usize).next_multiple_of(PAGE);
        let Some(shadow) = shadow_of(base, PAGE) else { return };
        shadow.writing(base, 16);
        shadow.writing(base + 128, 16);
        // SAFETY: these bytes belong to the test and its mapped copy.
        unsafe {
            (base as *mut u8).add(64).write(1);
            shadow.data.add(64).write(1); // mimic the middle range's refresh
        }
        shadow.writing(base + 64, 16);
        // SAFETY: the CPU changes the middle byte after the snapshot; an
        // actual GPU change is mimicked in the first range.
        unsafe {
            (base as *mut u8).add(64).write(2);
            shadow.data.add(3).write(9);
        }
        for at in [base, base + 128, base + 64] { shadow.copy_back(at, 16); }
        // SAFETY: still inside memory, which remains live for this test.
        unsafe {
            assert_eq!((base as *const u8).add(64).read(), 2, "CPU's newer byte");
            assert_eq!((base as *const u8).add(3).read(), 9, "GPU's actual write");
        }
    }

}
