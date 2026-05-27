//! Cranelift-based tier-2 JIT backend for QEMU TCG.
//!
//! The crate exposes a small C ABI (see `include/tcg/cranelift_bridge.h`)
//! that QEMU calls into when it wants to upgrade a hot TB from
//! TCG-emitted ARM64 to Cranelift-emitted ARM64.
//!
//! ```text
//!   x86 frontend                tcg_optimize           tier-2 dispatcher
//!        │                            │                       │
//!        ▼                            ▼                       ▼
//!     TCG IR  ─── tier-1 (existing) ──── ARM64                Cranelift
//!                                          │                       │
//!                                          └─── RCU swap ──────────┘
//! ```
//!
//! Module map:
//!
//! - [`ffi`]       C ABI surface; the only items linked from the C side.
//! - [`ir`]        Internal representation of the post-optimization TCG ops.
//! - [`translator`]  TCG IR -> Cranelift IR translator.
//! - [`memory`]    Guest-memory ops (qemu_ld / qemu_st) TLB fast-path.
//! - [`helper`]    TCG helper-call ABI marshalling.
//! - [`fpvec`]     FP / vector / atomic op lowering.
//! - [`dispatcher`]  Worker thread, request queue, RCU swap.
//! - [`telemetry`]  Counters surfaced through [`ffi::cranelift_tcg_get_stats`].
//! - [`tlb`]       TLB-entry layout the memory module needs to know.

#![allow(clippy::missing_safety_doc)]
#![allow(clippy::too_many_arguments)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod env;
pub mod ffi;
pub mod ir;
pub mod opc;
pub mod telemetry;
pub mod tlb;
pub mod translator;
pub mod memory;
pub mod helper;
pub mod fpvec;
pub mod x87;
pub mod dispatcher;
pub mod context;
pub mod verify;

#[cfg(test)]
mod tests;
