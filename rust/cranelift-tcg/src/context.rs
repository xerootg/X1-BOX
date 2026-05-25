//! Process-wide JIT context.
//!
//! Owns:
//! - The Cranelift `JITModule` (RWX-mapped code arena).
//! - The dispatcher worker thread and its request/response channels.
//! - The hot-block registry (keyed by guest TB PC).
//! - The crash blacklist (PC ranges that previously faulted).
//! - All shared telemetry.

use std::collections::{HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
    Arc,
};

use parking_lot::{Mutex, RwLock};

use crate::env::EnvDesc;
use crate::telemetry::Telemetry;

/// Number of MemOp shapes the helper arrays cover (size|sign|endian
/// bits).  Mirrors `qemu_ld_helpers[MO_SSIZE + 1]` and the padded
/// `qemu_st_helpers` on the C side.
pub const HELPER_ARRAY_LEN: usize = 16;

/// Snapshot of QEMU's softmmu slow-path helper tables.  Each entry is
/// the absolute address of a `helper_ld*_mmu` / `helper_st*_mmu`
/// function; NULL means the MemOp shape is not supported and tier-2
/// must bail to the slow path.
///
/// We store the pointers as `AtomicUsize` so the C side can publish the
/// arrays without having to hold a lock and the worker thread can read
/// them lock-free during compile.
pub struct HelperTable {
    pub ld: [AtomicUsize; HELPER_ARRAY_LEN],
    pub st: [AtomicUsize; HELPER_ARRAY_LEN],
}

impl HelperTable {
    pub fn empty() -> Self {
        // AtomicUsize isn't Copy, but `[expr; N]` accepts a const
        // initialiser; bind one and reuse it.
        const Z: AtomicUsize = AtomicUsize::new(0);
        HelperTable {
            ld: [Z; HELPER_ARRAY_LEN],
            st: [Z; HELPER_ARRAY_LEN],
        }
    }

    pub fn publish(&self, ld: &[usize; HELPER_ARRAY_LEN], st: &[usize; HELPER_ARRAY_LEN]) {
        for i in 0..HELPER_ARRAY_LEN {
            self.ld[i].store(ld[i], Ordering::Release);
            self.st[i].store(st[i], Ordering::Release);
        }
    }

    pub fn ld_helper(&self, memop: u32) -> Option<usize> {
        let idx = (memop & 0x0f) as usize;
        let v = self.ld[idx].load(Ordering::Acquire);
        if v == 0 { None } else { Some(v) }
    }

    pub fn st_helper(&self, memop: u32) -> Option<usize> {
        // Stores have no sign bit; mask to size only.
        let idx = (memop & 0x07) as usize;
        let v = self.st[idx].load(Ordering::Acquire);
        if v == 0 { None } else { Some(v) }
    }
}

/// One blacklisted guest-PC range.
#[derive(Copy, Clone, Debug)]
pub struct BlacklistRange {
    pub lo: u64,
    pub hi: u64,
}

/// Result of a successful tier-2 compile.
#[derive(Clone, Debug)]
pub struct TierTwoEntry {
    /// Pointer to the emitted code (lives in the JIT arena).
    pub code: *const u8,
    /// Code size in bytes.
    pub size: usize,
    /// Guest PC the TB starts at.
    pub tb_pc: u64,
}

// SAFETY: TierTwoEntry only holds a raw pointer into an mmap'd arena
// that lives as long as the JitContext. The pointer is never deref'd
// off the JIT thread.
unsafe impl Send for TierTwoEntry {}
unsafe impl Sync for TierTwoEntry {}

/// Configuration knobs (all atomically settable from the C side).
pub struct Config {
    pub enabled: AtomicBool,
    pub verify_mode: AtomicBool,
    pub hot_threshold: AtomicU32,
    pub max_entries: AtomicU32,
}

impl Config {
    pub fn defaults() -> Self {
        Config {
            enabled: AtomicBool::new(true),
            verify_mode: AtomicBool::new(false),
            hot_threshold: AtomicU32::new(1000),
            // 4096 was too small for Halo 2 (saturates instantly, then LRU
            // thrash on every new hot TB). 32K covers the working set with
            // headroom and costs ~10MB of compiled Cranelift code.
            max_entries: AtomicU32::new(32768),
        }
    }
}

/// Per-process JIT state.
pub struct JitContext {
    pub env: EnvDesc,
    pub config: Config,
    pub stats: Arc<Telemetry>,
    pub blacklist: RwLock<Vec<BlacklistRange>>,
    pub entries: RwLock<HashMap<u64, TierTwoEntry>>,
    pub lru: Mutex<VecDeque<u64>>,
    pub dispatcher: Mutex<Option<crate::dispatcher::Dispatcher>>,
    pub helpers: HelperTable,
}

impl JitContext {
    pub fn new(env: EnvDesc) -> Arc<Self> {
        let stats = Arc::new(Telemetry::new());
        let ctx = Arc::new(Self {
            env,
            config: Config::defaults(),
            stats: Arc::clone(&stats),
            blacklist: RwLock::new(Vec::new()),
            entries: RwLock::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            dispatcher: Mutex::new(None),
            helpers: HelperTable::empty(),
        });

        let dispatcher = crate::dispatcher::Dispatcher::spawn(Arc::clone(&ctx));
        *ctx.dispatcher.lock() = Some(dispatcher);
        ctx
    }

    pub fn is_blacklisted(&self, pc: u64) -> bool {
        self.blacklist
            .read()
            .iter()
            .any(|r| pc >= r.lo && pc < r.hi)
    }

    pub fn record_blacklist(&self, lo: u64, hi: u64) {
        self.blacklist.write().push(BlacklistRange { lo, hi });
        self.stats.blacklisted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn insert_entry(&self, entry: TierTwoEntry) {
        let pc = entry.tb_pc;
        let cap = self.config.max_entries.load(Ordering::Relaxed) as usize;
        let mut entries = self.entries.write();
        let mut lru = self.lru.lock();
        if let Some(existing) = entries.insert(pc, entry.clone()) {
            // overwrite: re-use existing LRU slot
            let _ = existing;
            if let Some(pos) = lru.iter().position(|&p| p == pc) {
                lru.remove(pos);
            }
        }
        lru.push_back(pc);
        while entries.len() > cap {
            if let Some(victim) = lru.pop_front() {
                entries.remove(&victim);
            } else {
                break;
            }
        }
        self.stats
            .active_entries
            .store(entries.len() as u32, Ordering::Relaxed);
    }

    pub fn lookup(&self, pc: u64) -> Option<TierTwoEntry> {
        self.entries.read().get(&pc).cloned()
    }

    /// Drop every cached tier-2 entry. Called from the C bridge on a
    /// QEMU `tb_flush` because all keys (TranslationBlock pointers) are
    /// about to alias completely different guest TBs.
    pub fn reset_entries(&self) {
        let mut entries = self.entries.write();
        let mut lru = self.lru.lock();
        entries.clear();
        lru.clear();
        self.stats.active_entries.store(0, Ordering::Relaxed);
    }

    pub fn shutdown(&self) {
        if let Some(dispatcher) = self.dispatcher.lock().take() {
            dispatcher.shutdown();
        }
    }
}

impl Drop for JitContext {
    fn drop(&mut self) {
        self.shutdown();
    }
}
