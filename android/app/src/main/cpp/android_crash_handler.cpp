#include <android/log.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <unwind.h>

namespace {
constexpr const char* kCrashTag = "xemu-android";
constexpr size_t kPathMax = 512;

static char g_inline_aio_flag_path[kPathMax];

static int GetTid() {
  return static_cast<int>(syscall(SYS_gettid));
}

struct BacktraceState {
  void** addrs;
  int count;
  int max;
};

static _Unwind_Reason_Code UnwindCallback(struct _Unwind_Context* ctx, void* arg) {
  BacktraceState* state = static_cast<BacktraceState*>(arg);
  if (state->count >= state->max) {
    return _URC_END_OF_STACK;
  }
  uintptr_t pc = _Unwind_GetIP(ctx);
  if (pc != 0) {
    state->addrs[state->count++] = reinterpret_cast<void*>(pc);
  }
  return _URC_NO_REASON;
}

static void LogBacktrace() {
  void* addrs[64];
  BacktraceState state{addrs, 0, static_cast<int>(sizeof(addrs) / sizeof(addrs[0]))};
  _Unwind_Backtrace(UnwindCallback, &state);
  for (int i = 0; i < state.count; ++i) {
    Dl_info info;
    if (dladdr(addrs[i], &info) && info.dli_fname) {
      uintptr_t base = reinterpret_cast<uintptr_t>(info.dli_fbase);
      uintptr_t pc = reinterpret_cast<uintptr_t>(addrs[i]);
      uintptr_t rel = (base != 0 && pc >= base) ? (pc - base) : 0;
      const char* sym = info.dli_sname ? info.dli_sname : "?";
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  #%02d pc %p %s (%s+0x%zx)",
                          i, addrs[i], info.dli_fname, sym,
                          static_cast<size_t>(rel));
    } else {
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag, "  #%02d pc %p", i, addrs[i]);
    }
  }
}

#if defined(__aarch64__)
/*
 * Walk the AArch64 frame-pointer chain by hand. Each frame's x29 (FP)
 * points to [saved FP, saved LR] at the top of the previous frame.
 * Much more robust than _Unwind_Backtrace when crossing libc internals
 * or signal frames — libunwind frequently bails out on those.
 */
static void LogFpChain(uintptr_t fp, uintptr_t lr) {
  /*
   * Frame 0 is the current pc (already logged by the caller). We start
   * by logging the lr we extracted from ucontext (frame 0's return
   * site), then walk the FP chain.
   */
  for (int i = 0; i < 32 && fp != 0; ++i) {
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(lr), &info) && info.dli_fname) {
      uintptr_t base = reinterpret_cast<uintptr_t>(info.dli_fbase);
      uintptr_t rel = (base != 0 && lr >= base) ? (lr - base) : 0;
      const char* sym = info.dli_sname ? info.dli_sname : "?";
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  fp#%02d lr=%p %s (%s+0x%zx) fp=%p",
                          i, (void *)lr, info.dli_fname, sym,
                          static_cast<size_t>(rel), (void *)fp);
    } else {
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  fp#%02d lr=%p <unknown> fp=%p",
                          i, (void *)lr, (void *)fp);
    }
    // Frame format on AArch64: [previous FP, return LR]. Both 8 bytes.
    // Read from FP+0 and FP+8. Bail if FP looks bogus to avoid SIGSEGV
    // inside the handler itself.
    if (fp < 0x10000 || (fp & 7) != 0) break;
    uintptr_t prev_fp = *reinterpret_cast<uintptr_t volatile *>(fp);
    uintptr_t prev_lr = *reinterpret_cast<uintptr_t volatile *>(fp + 8);
    if (prev_fp == 0 || prev_fp <= fp) break;
    fp = prev_fp;
    lr = prev_lr;
  }
}
#endif

static void MarkInlineAioRequired() {
  if (g_inline_aio_flag_path[0] == '\0') {
    return;
  }

  int fd = open(g_inline_aio_flag_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    return;
  }

  static const char kValue[] = "1\n";
  ssize_t ignored = write(fd, kValue, sizeof(kValue) - 1);
  (void)ignored;
  close(fd);
}

static void CrashHandler(int sig, siginfo_t* info, void* ucontext) {
  if (sig == SIGILL) {
    MarkInlineAioRequired();
  }
  uintptr_t pc = 0;
  uintptr_t lr = 0;
  uintptr_t sp = 0;
  uintptr_t fp = 0;
  uintptr_t fault_addr = info ? reinterpret_cast<uintptr_t>(info->si_addr) : 0;
#if defined(__aarch64__)
  if (ucontext) {
    auto* uc = static_cast<ucontext_t*>(ucontext);
    pc = uc->uc_mcontext.pc;
    sp = uc->uc_mcontext.sp;
    fp = uc->uc_mcontext.regs[29];
    lr = uc->uc_mcontext.regs[30];
  }
#endif
  __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                      "Caught signal %d in tid %d pc=0x%016zx lr=0x%016zx "
                      "sp=0x%016zx fp=0x%016zx fault=0x%016zx",
                      sig, GetTid(),
                      static_cast<size_t>(pc), static_cast<size_t>(lr),
                      static_cast<size_t>(sp), static_cast<size_t>(fp),
                      static_cast<size_t>(fault_addr));
  // Resolve pc/lr through dladdr so we can see which .so they live in.
  if (pc != 0) {
    Dl_info dinfo;
    if (dladdr(reinterpret_cast<void*>(pc), &dinfo) && dinfo.dli_fname) {
      uintptr_t base = reinterpret_cast<uintptr_t>(dinfo.dli_fbase);
      uintptr_t rel = (pc >= base) ? (pc - base) : 0;
      const char* sym = dinfo.dli_sname ? dinfo.dli_sname : "?";
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  pc -> %s (%s+0x%zx)",
                          dinfo.dli_fname, sym,
                          static_cast<size_t>(rel));
    } else {
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  pc -> <unknown mapping>");
    }
  }
  if (lr != 0) {
    Dl_info dinfo;
    if (dladdr(reinterpret_cast<void*>(lr), &dinfo) && dinfo.dli_fname) {
      uintptr_t base = reinterpret_cast<uintptr_t>(dinfo.dli_fbase);
      uintptr_t rel = (lr >= base) ? (lr - base) : 0;
      const char* sym = dinfo.dli_sname ? dinfo.dli_sname : "?";
      __android_log_print(ANDROID_LOG_ERROR, kCrashTag,
                          "  lr -> %s (%s+0x%zx)",
                          dinfo.dli_fname, sym,
                          static_cast<size_t>(rel));
    }
  }
  LogBacktrace();
#if defined(__aarch64__)
  LogFpChain(fp, lr);
#endif
  signal(sig, SIG_DFL);
  raise(sig);
}

static void InstallCrashHandlers() {
  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_sigaction = CrashHandler;
  sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
}
}  // namespace

extern "C" void xemu_android_set_inline_aio_crash_flag_path(const char* path) {
  if (!path) {
    g_inline_aio_flag_path[0] = '\0';
    return;
  }

  size_t len = strnlen(path, sizeof(g_inline_aio_flag_path) - 1);
  memcpy(g_inline_aio_flag_path, path, len);
  g_inline_aio_flag_path[len] = '\0';
}

__attribute__((constructor)) static void InstallCrashHandlersOnLoad() {
  InstallCrashHandlers();
}
