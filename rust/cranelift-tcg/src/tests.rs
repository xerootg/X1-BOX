//! Unit tests. Only built when `cargo test` is run on the host.
//!
//! These exercise the translator with hand-crafted TCG IR. They do NOT
//! execute the emitted code (that would need a JIT-runnable host with
//! the same env layout); they verify that the compile pipeline produces
//! verified Cranelift IR without errors.

#![cfg(test)]

use crate::env::EnvDesc;
use crate::ffi::{cranelift_tcg_compile_sync, cranelift_tcg_init, CraneliftTcgEnvDesc};
use crate::ir::RawOp;
use crate::opc::Opc;

fn mk_op(opc: Opc, args: &[u64], oargs: u8, iargs: u8, cargs: u8) -> RawOp {
    let mut a = [0u64; 16];
    for (i, v) in args.iter().enumerate().take(16) {
        a[i] = *v;
    }
    RawOp {
        opc: opc as u16,
        nb_oargs: oargs,
        nb_iargs: iargs,
        nb_cargs: cargs,
        type_: 1, // I64
        flags: crate::opc::flags::TCG_OPF_INT,
        const_mask: 0,
        _pad: 0,
        args: a,
    }
}

unsafe fn init_with_dummy() -> *mut std::ffi::c_void {
    // SAFETY: this is a zero-initialised env desc — every pointer is
    // null, every offset is 0. The tests below exercise tier-2
    // compile paths that don't touch the env layout or the new x87
    // surface, so zeroing is fine. Using std::mem::zeroed() avoids
    // having to spell out every new field whenever the struct grows.
    let desc: CraneliftTcgEnvDesc = unsafe { std::mem::zeroed() };
    let mut desc = desc;
    desc.env_size = 0x4000;
    desc.guest_ptr_size = 4;
    desc.host_ptr_size = 8;
    unsafe { cranelift_tcg_init(&desc) }
}

#[test]
fn compile_exit_tb_only() {
    // Minimal TB: just exit. Should produce a function returning 0.
    let ops = vec![mk_op(Opc::ExitTb, &[0], 0, 0, 1)];
    let ctx = unsafe { init_with_dummy() };
    assert!(!ctx.is_null());

    let mut code: *const std::ffi::c_void = std::ptr::null();
    let mut size: usize = 0;
    let rc = unsafe {
        cranelift_tcg_compile_sync(
            ctx,
            0xdead_beef,
            ops.as_ptr(),
            ops.len(),
            &mut code,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "exit_tb compile should succeed");
    assert!(!code.is_null(), "code pointer must be set");
    assert!(size > 0, "code size must be non-zero");
}

#[test]
fn compile_add_then_exit() {
    // out = a + b ; exit 0
    let ops = vec![
        // mov tmp0, 10
        mk_op(Opc::Mov, &[/*o*/ 0, /*i*/ 1], 1, 1, 0),
        // mov tmp1, 20
        mk_op(Opc::Mov, &[/*o*/ 2, /*i*/ 3], 1, 1, 0),
        // tmp4 = tmp0 + tmp1
        mk_op(Opc::Add, &[/*o*/ 4, /*i*/ 0, 2], 1, 2, 0),
        // exit_tb 0
        mk_op(Opc::ExitTb, &[0], 0, 0, 1),
    ];
    let ctx = unsafe { init_with_dummy() };
    assert!(!ctx.is_null());

    let mut code: *const std::ffi::c_void = std::ptr::null();
    let mut size: usize = 0;
    let rc = unsafe {
        cranelift_tcg_compile_sync(
            ctx,
            0xc0de_0001,
            ops.as_ptr(),
            ops.len(),
            &mut code,
            &mut size,
        )
    };
    assert_eq!(rc, 0, "add+exit compile should succeed");
}

#[test]
fn env_desc_dummy_roundtrip() {
    let d = EnvDesc::dummy();
    assert_eq!(d.host_ptr_size, 8);
    assert_eq!(d.guest_ptr_size, 4);
}

#[test]
fn raw_op_layout_is_144() {
    // const_mask + _pad bumped RawOp from 136 to 144 bytes; mirror the
    // const assertion in src/ir.rs at runtime so a future shrink that
    // breaks the C ABI shows up here as well.
    assert_eq!(core::mem::size_of::<RawOp>(), 144);
}
