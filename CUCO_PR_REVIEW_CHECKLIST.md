# cudax::cuco PR review checklist (distilled from NVIDIA/cccl review history)

Last updated: **2026-11-13**.

Source: review history of about 60 cuco-related NVIDIA/cccl PRs (#4898 ... #12442): hashers,
HyperLogLog, bloom_filter, fixed_capacity_map/set, atomics, infra. Items marked **(recurring)**
arose in two or more PRs. Bot/CI noise is excluded; an unresolved bot finding is not an accepted
rule. Where a convention changed, the current rule is stated.

---

## 1. Annotations and visibility macros

- **(recurring)** Every function in a cudax/cuco header gets `_CCCL_HOST_API`, `_CCCL_DEVICE_API`
  or `_CCCL_HOST_DEVICE_API` (constructors, `operator()`, static members, free helpers, strong-type
  wrappers, kernels' device helpers). Never raw `__host__ __device__`, never bare `_CCCL_HOST` /
  `_CCCL_DEVICE` / `_CCCL_HOST_DEVICE`. (#4898, #6666, #7705,
  #8857, #9675, #10695/#10759/#10888)
- **Current rule: do not use `_CCCL_API` in cudax.** It carries `_CCCL_TILE`, unusable in tile
  mode; cudax moved off it in #8955. Older cuco PRs (#4898, #5400) used `_CCCL_API`; do not copy
  them. libcu++ code (e.g. `cuda::hash`) still uses `_CCCL_API`. (#7705, #9675)
- **(recurring)** Tests and benchmarks also use the `_CCCL_*_API` macros for functors and kernels
  helpers, not `__host__ __device__`. (#10218, #10740, #10695)
- Mark `= default` special members `_CCCL_HIDE_FROM_ABI`. (#9720, #11345)
- Attributes go first: `[[nodiscard]] _CCCL_HOST_DEVICE_API ...`. (#4898, #5400)
- **(recurring)** `[[nodiscard]]` on every getter, pure computation, factory and call operator with
  a non-void return. (#4898, #5400, #6666 x many, #10757)
- Host-only owning classes (anything holding a `cuda::buffer`): declare the destructor explicitly
  `_CCCL_HOST_API ~T() {} // NOLINT(modernize-use-equals-default)` (nvcc ignores the host
  annotation on a defaulted dtor and emits diag 20011), then explicitly `= default` copy/move
  ctors and move-assign, and `static_assert` the resulting traits in a test. Never paper over with
  `#pragma nv_diag_suppress 20011`. (#10694 -> #11017, #11345, #10740)
- Any remaining diagnostic suppression needs a comment explaining the cause, a compiler guard, and
  a tracking issue link. Reviewers will ask "what is the issue here?". (#10694,
  #10757)
- Use `NV_IF_TARGET` / `NV_IF_ELSE_TARGET` for host/device branches, not `#ifdef __CUDA_ARCH__`;
  and re-verify branch polarity when converting (`back_to_back_cas` needs ITS, so it is the SM70+
  branch). (#6666, #9877)

## 2. Config header, include guards, includes, file layout

- **Current rule:** `#include <cuda/std/detail/__config>` immediately after the include guard in
  every cuco header, including `detail/**`. `<cuda/__cccl_config>` is superseded (#4898 era).
  Tests get the macros via `<cuda/__cccl_config>`. Never include anything under
  `cuda/std/__cccl/` directly. (#7705, #9675, #10695/#10757)
- Every header: `<cuda/std/__cccl/prologue.h>` as the last include before code,
  `<cuda/std/__cccl/epilogue.h>` at the end. (#7705)
- All includes except the config header come after the system-header pragma block. (#4898)
- **(recurring)** Include exactly what you use, with the fine-grained header
  (`<cuda/std/__cstddef/types.h>`, `<cuda/std/__fwd/span.h>`, `<cuda/__functional/hash.h>`,
  `<cuda/__container/buffer.h>`, `<cuda/__utility/no_init.h>`, `<cuda/std/__tuple_dir/tie.h>`
  for `make_tuple`). No umbrella headers except `<cuda/std/cstdint>`; remove unused and duplicate
  includes; verify the fine-grained header actually exists before "fixing" an include.
  (#4898, #6666 x8, #7705, #10521/#10695/#10757, #10621)
- **(recurring)** Never plain `<cstdint>`, `<vector>` or other std headers in headers; use
  `cuda::std` counterparts and `cuda::buffer` for host temporaries. (#4898, #5558, #6666)
- Tests include what they use directly (`<cuda/functional>`, `<vector_types.h>` for `char3`,
  `"test_macros.h"`), no transitive reliance. (#10621, #10025)
- Inspect include-cleaner findings before deleting dependencies, especially uses inside
  `NV_IF_TARGET`; prefer precise subheaders and central export/keep annotations over repeated
  consumer suppressions. The merged hash cleanup removed unused `always_false.h` / `cstddef`
  includes; it did not enable the check across CCCL. (#11400)
- Include guards mirror the path: `_CUDAX___CUCO_DETAIL_HASH_FUNCTIONS_MURMURHASH3_CUH`; renaming or
  moving a file means renaming the guard and its `#endif` comment. Short form, not the full
  `_CUDAX_INCLUDE_...` path. (#4898, #6666, #7705, #9436, #9735)
- Layout: public headers at `__cuco/*.cuh`, implementation under `__cuco/detail/<component>/`.
  Do not `__`-prefix a subdirectory that already lives under a `__` parent (`__cuco/__detail` is
  wrong). A file in `detail/` must not declare public-namespace entities; user-facing template
  arguments (`conditional_add_mode`, `empty_key`, policies) live in public headers, with generic
  tuning-mode names grouped in a struct or namespace to avoid collisions.
  (#7705, #9436, #10740)
- Do not create a header for a single tiny helper unless several TUs share it. (#11449)
- **All cudax headers are `.cuh`, including host-only utilities.** cuDF's `.hpp`-means-host-safe
  convention is not adopted. Do not widen the header-test glob; rename instead. (#9696/#9735)
- License header: SPDX form. Library headers use the "Part of CUDA Experimental in CUDA C++ Core
  Libraries … Apache-2.0 WITH LLVM-exception" block; tests/benchmarks the two-line SPDX form.
  Update the copyright year on every new file (asked three times in one PR). Third-party licensed
  code (xxHash BSD-2) gets an SPDX notice at the top and a lead sign-off. (#4898, #5558,
  #6666)

## 3. Naming and namespaces

- **(recurring)** Uglify every non-public identifier: `__name` for variables, members, functions,
  kernels, aliases, namespaces; `_Ugly` for template parameters (`_Tp`, not `_T`). One leading
  `__`, never `____`. Public entry points must NOT be ugly (an intended-public `__compute_hash`
  had to be exposed; `empty_key` etc. are unprefixed). (#4898 x4, #6666, #7705)
- **(recurring)** Tests and benchmarks are user code: no `__ugly` names, no `::` leading colons
  required, and no calls into `detail`/`__` APIs. (#9748, #9868)
- `snake_case` everywhere, including implementation classes (`__hyperloglog_impl`, not
  `HyperLogLogImpl`) and public types (`xxhash_32`, `hash`). (#5400, #6666)
- **(recurring)** Fully qualify every call into other namespaces in library headers:
  `::cuda::std::memcpy`, `::cuda::std::rotl`, `::cooperative_groups::…`, `CUB_NS_QUALIFIER::`;
  never rely on ADL or `using`. `_CUDA_VSTD::` macros are dropped, spell `::cuda::std::`.
  But do NOT repeat the enclosing namespace for your own symbols (`::cuda::experimental::cuco::`
  inside `cuco` is noise). (#4898, #5400 x3, #6666, #7705, #10740)
- No `this->` unless a dependent-base lookup requires it. (#7705, #9720, #10757/#10759)
- Accessors have no `get_` prefix (`policy()`); scope-rebinding alias is `rebind_scope`; keep the
  same alias name across cooperating types; prefer established abbreviations (`hllpp_finalizer`).
  (#8857)
- Only one spelling of a trait: `is_foo_v` variable template (or class as `bool_constant`); keep
  sibling trait names consistent (`is_pair_like` next to `is_tuple_like`). (#7705)
- Bikeshed names early, before the rest of the migration: `static_map` -> `fixed_capacity_map`,
  `strategy` -> `hash_algorithm`. (#5400, #7705)

## 4. Feature macros, compiler and toolkit checks

- **(recurring)** Feature macros are function-like and always 0/1: `#define _CCCL_CUCO_HAS_X() 1`;
  never object-like `#define FOO` + `#ifdef`. Private macros carry the `_CCCL_CUCO_` prefix, not
  `CUCO_` (users may include both libraries). (#4898 x4)
- Use CCCL helpers, never hand-rolled checks: `_CCCL_CTK_AT_LEAST(12,1)` / `_CCCL_CTK_BELOW`,
  `_CCCL_PTX_ARCH()`, `_CCCL_HAS_INT128()`, `_CCCL_CUDA_COMPILER(NVCC)`, `_CCCL_COMPILER(NVRTC)`,
  `_CCCL_CUDA_COMPILATION()`. Drop CUDA 11 fallbacks; CCCL supports CTK >= 12.0. (#4898, #6666)
- **(recurring)** Every `#else` / `#endif` carries the annotated condition:
  `#else // ^^^ COND ^^^ / vvv !COND vvv`, `#endif // _CCCL_HAS_INT128()`. Repeat the verbatim
  `#if` condition. (#4898, #5400, #6666, #7705, #10621)
- **(recurring)** Guard for the three compilation modes: host-only implementation (owning classes,
  launches, occupancy, runtime/driver calls, `copy_bytes`) in `#if !_CCCL_COMPILER(NVRTC)`;
  kernel definitions and launches in `#if _CCCL_CUDA_COMPILATION() && !_CCCL_COMPILER(NVRTC)` so a
  plain `.cpp` TU can include the public header. Do NOT use `_CCCL_HOST_COMPILATION()` for this
  (false in nvcc's device pass). Verify host-only downstream consumers (RAPIDS) still see the
  public type. (#7705, #9720, #10558, #10693/#10757 CI, #10740)
- Don't `#error` on non-nvcc CUDA compilers; guard or warn. (#4898)
- Prefer disabling an unsupported-compiler test in CMake over sprinkling `#if` in test sources;
  every `UNSUPPORTED`/disabled test needs a comment with the defect and a removal criterion.
  (#9076/#9185, #10621)
- Don't guard *enumerators* with feature macros; `static_assert` inside the specialization
  instead so the public enum shape doesn't depend on flags. (#10621)
- Check that macros you port still exist on `main`: `_CCCL_TRY_CUDA_API` was split into
  `_CCCL_TRY_RUNTIME_API` / `_CCCL_TRY_DRIVER_API` (#11177); `hash_functions.cuh` is deprecated in
  favour of `<cuda/__functional/hash.h>` / `cuda::hash<Key, hash_algorithm::xxhash_64>`.
  (#6666, #10740)
- Don't add extra nvcc flags (`--extended-lambda`) or leftover per-test CMake settings without a
  concrete need. (#6666)

## 5. noexcept, constexpr, exceptions, assertions

- **(recurring, top offender)** Never `noexcept` on anything that can throw: every function that
  calls a CUDA runtime/driver API, launches a kernel (sync and `_async` wrappers alike), or
  invokes user-supplied functors/hashers/key-equal/sentinel types (constructors, `key_eq()`,
  `hash_function()`, `rebind_*`, `ref()`, `make_iterator`, strong-type conversions). Audit every
  `noexcept` inherited from cuCollections; a launch failure through a `noexcept` wrapper
  terminates the process. (#7705, #10757, #10888, #11391)
- Conversely, DO keep `noexcept` (and `constexpr`) on pure computation ported from cuco: hasher
  `operator()`, device members that only call noexcept policy functions/shuffles/atomics, and
  their public ref wrappers. Dropping them wholesale is a regression. (#4898, #5400, #10740)
- Check object construction/copying hidden in value-returning getters: returning a custom hasher
  or comparator can throw even when the getter contains no CUDA call. In #11391 the author
  deliberately removed `noexcept` from affected strong-type operations rather than maintaining
  conditional specifications; the bot withdrew its contrary request. This is the affected
  code's policy, not a ban on conditional `noexcept` throughout CCCL.
  ([Getter explanation](https://github.com/NVIDIA/cccl/pull/11391#discussion_r4168030050),
  [strong-type decision](https://github.com/NVIDIA/cccl/pull/11391#discussion_r4123876013); merged October 5)
- Only `constexpr` what can be constant-evaluated: not functions doing `memcpy` through pointers,
  `reinterpret_cast`, `log`/`round`. Add `constexpr` where possible (finalizers, size math).
  (#4898, #6666, #10740)
- **(recurring)** Wrap CUDA API calls in `_CCCL_TRY_RUNTIME_API(::cudaFoo, "cuco: msg", args…)`
  (driver: `_CCCL_TRY_DRIVER_API`); failures throw `::cuda::cuda_error`. Never `_CCCL_ASSERT` for
  runtime failures (stripped in release), never hand-checked return codes. (#7705, #10888, #11391)
- **(recurring)** Check every raw `<<<>>>` launch with `_CCCL_TRY_RUNTIME_API(::cudaGetLastError,…)`
  — or better, don't raw-launch at all (see §7). Adding checks to previously unchecked paths must
  come with removing `noexcept` in the same PR. (#10693, #10740, #10888)
- Bad user arguments: `_CCCL_THROW(::std::invalid_argument, "msg")`; `__throw_cuda_error` is for
  CUDA failures only. Don't `_CCCL_ASSERT` right before throwing the same condition. Document
  `@throws` on the public API, not only the impl. (#6666, #7705, #10888)
- `_CCCL_ASSERT` preconditions with both lower and upper bounds (`cuda::in_range`); device code
  that can't throw must at least assert (it traps, not UB). Add it the first time, not the third.
  (#6666 x3)
- Audit the exception guarantee after every ownership change: #11391's rehash swaps storage
  before constructing a potentially throwing `ref()`, with no rollback. Moving `ref()` before the
  swap would capture the old slots, so that alone is not a fix. (#11391)
- Non-owning `*_ref` constructors validate caller-provided storage and throw instead of rounding
  down silently; test undersized and rounded-down inputs. (#10220)
- Overflow: `cuda::add_overflow` / `cuda::mul_overflow`, and check *after* `next_prime` enlarges
  the operand; guard shifts against full-width counts (`1u << 32` is UB); cast explicitly when
  comparing `uint64_t` max against `double`. (#7705)
- `static_assert` the real constraints, delete dead ones: hasher return is unsigned and wide
  enough; `_BucketSize != 0`; supported packed widths; remove asserts on facilities no longer
  used (`atomic_ref::fetch_or`). (#6666, #7705, #8855, #10025, #10740)

## 6. Streams and memory resources

- **(recurring, convention)** No default stream, no default memory resource anywhere. Constructors
  take `cuda::stream_ref` first, memory resource second (matching `cuda::buffer`); every host API
  including `_async` variants takes the stream as first parameter with no default. Never default to
  `device_default_memory_pool(device_ref{0})`. (#7705, #9719, #9720)
- Do not store the construction stream in the container; the stream is per-operation.
  (#7705)
- CCCL runtime has no "current device": helpers take an explicit `cuda::device_ref`, never call
  `cudaGetDevice`; delete unused helpers. Use fully-qualified enumerators
  (`::cudaDevAttrMaxSharedMemoryPerBlockOptin`) and prefer `::cuda::__driver` / runtime C++
  wrappers over raw calls. (#6666, #7705)
- For stream-bound setup, obtain attributes from that stream's device, e.g.
  `cuda::device_attributes::max_shared_memory_per_block_optin(stream.device())`.
  Audit context-sensitive setup before launch as well as the launch itself; #11391's context
  guard and restoration test landed (a review thread on it stayed unresolved, so do not read it as an API recommendation).
  ([#11391 human review](https://github.com/NVIDIA/cccl/pull/11391#discussion_r4149918862))
- **(recurring)** All temporaries (counters, temp storage) through the given MR via
  `cuda::device_buffer` / `cuda::buffer`, constructed directly with the initial value
  (`device_buffer<T>{stream, mr, 1, T{0}}`); never `cudaMalloc`/`cudaFree` or allocate-then-memset.
  Use `atomic_ref` on the element, never `reinterpret_cast` to `cuda::std::atomic`. (#6666, #7705)
- D2H readbacks: `cuda::copy_bytes` from `<cuda/__algorithm/copy.h>` with a host `cuda::std::span`;
  no cuco-local wrapper, no `__driver::__memcpyAsync`. Guard under `!NVRTC`; verify on oldest CI
  toolchain (CTK 12.0 + MSVC crashed). (#10558)
- `::cuda::no_init` needs `<cuda/__utility/no_init.h>` and the `::cuda::no_init_t` type. (#6666, #9720)
- Constant lookup tables: not `__constant__` (pollutes TUs, crashes compilers), not `static inline`;
  use `_CCCL_GLOBAL_CONSTANT cuda::std::array<…>` or pass as a `__grid_constant__` parameter.
  (#6666)
- Prefer the CUB single-phase env-based API (`env{stream, mr}`) over two-phase temp storage.
  (#10521)
- **(recurring)** Tests: one dedicated `cuda::stream`, all work on it (`thrust::cuda::par.on(...)`
  or `cuda::buffer`), `stream.sync()` not `cudaDeviceSynchronize()`, `cuda::buffer` instead of
  `thrust::device_vector`; never mix in default-stream Thrust calls. (#9719 x9)

## 7. Kernels, launch, cooperative groups, device code

- **(recurring)** Launch with `cuda::launch(stream, ::cuda::make_config(grid_dims, block_dims),
  kernel, args…)` — no raw `<<<>>>`, in library and tests. Configure dynamic shared memory on
  the exact function/context used for launch before querying occupancy. #11391's merged HLL
  implementation uses the driver-side function attribute through `__get_cufunction_of` and
  `__functionSetAttributeNoThrow`, checked by `_CCCL_TRY_DRIVER_API`; do not copy its superseded
  runtime-attribute sketch. (#10740, #11095, #11399, #11391)
- Use compile-time launch dimensions when values are known (`grid_dims<1>()`,
  `block_dims<1024>()`); infer a constexpr block size from the thread-hierarchy object instead of
  passing it to the kernel separately. Check template-only local constants for nvcc/MSVC unused-variable
  diagnostics; inline them or use narrowly justified `[[maybe_unused]]`.
  ([#11391 review](https://github.com/NVIDIA/cccl/pull/11391#discussion_r4144993806), merged diff)
- In probing loops, an overloaded `break` inside nested loops/switches is a known hazard: when
  it must exit two levels, restructure or accept `goto` rather than add flag variables. (#11975)
- All `__global__` kernels need `__launch_bounds__`; cast grid size explicitly; no magic warp-size
  numbers. (#7705)
- Prefer CUB / hierarchy APIs to hand-rolled kernels and index helpers where they are drop-ins
  (`cub::DeviceFor`, `cub::DeviceTransform::Fill`, `cuda::hierarchy`); shared helpers
  (`__global_thread_id()`, `__is_full_tile()`) live in `detail/utility/cuda.cuh`. (#6666, #7705, #10740)
- Global thread id: `static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x` (cast an operand,
  not the result). Grid-stride loops written plainly. Cast `threadIdx.x` once when comparing to a
  signed bound. (#6666, #9738)
- Device API shape: single-thread overload + cooperative-group overload (and increasingly a
  `cg::thread_block` overload). Be ready to justify "do we really need CG?" (interface clarity:
  CG signals the impl may sync). When a group-generic overload replaces `thread_block_tile<cg_size>`,
  statically check the group size matches `cg_size`. (#6666, #9877)
- Preserve cuco perf optimizations when porting (buffer-flush strategy for cg_size 1,
  `cuda::memcpy_async` fast path in `make_copy`, `static_for`/`#pragma unroll`). If you drop one,
  say so and attach benchmarks. (#7705, #10621, #10688)
- Loop hygiene: fixed-trip loops via `_CCCL_PRAGMA_UNROLL_FULL()` / `::cuda::static_for`, not
  hand-unrolled sequences; no range-based `for` in hot loops; `_CCCL_BUILTIN_EXPECT(cond, 0)` on
  rare branches (probe wraparound); evaluate cheap predicates before probing in `*_if` kernels;
  shuffle indices, then form pointers (never form a pointer from an invalid index). Measure before
  adopting `static_for` in probing loops (was neutral to 3.8% slower). (#4898, #5400, #6666,
  #9868)
- Hash-to-index: narrow the hash to the unsigned size type BEFORE `%`, or a 128-bit hasher does a
  128-bit remainder (152 vs 128 instructions, 22 vs 18 regs). Use `cuda::fast_mod_div` only for
  dynamic extent, precomputed once at construction. Containers fix the extent to `size_t`, so
  take/return `size_t`, require an unsigned hash, and `static_assert` the index type in public
  entry points such as `make_iterator` instead of supporting 32-bit sizes. Keep one code path: a
  special case for a narrower hash or index type needs a measured win in the real find/insert
  kernels, not an isolated SASS count. (#7705, #11449, #12360)
- Shared memory: init smem atomics with `.store(0)`; no `__shared__ T buf[N]` with non-trivial `T`
  (Clang CUDA rejects it) — use `::cuda::__uninitialized_array`; objects in raw smem storage need
  an explicit destroy plus a group sync before rank 0 destroys them. (#6666, #10688, #10693)
- Distinguish trivial copying from trivial default construction in shared output staging.
  #11340 uses `cuda::__uninitialized_array` with explicit construction to support trivially
  copyable mapped types whose default constructor is nontrivial; preserve that case when
  simplifying temporary storage. (#11340)
- Take the key by value in bulk kernels (`const Key& k = *(first + idx)` binds to a temporary for
  by-value iterators). (#10740)
- In `insert_and_find`, wait for payload publication on paths where the key can appear before
  the mapped value. Choose the wait from the actual insertion path: a packable type with
  insufficient runtime alignment still uses the key-first fallback. Check both existing-key
  and lost-CAS paths, scalar and cooperative variants. The merged helper handles this runtime
  decision. Whenever a payload is published to other threads, ask what orders it against the
  readiness flag: the merged code keeps relaxed payload operations, and a bot's request for
  release/acquire was never adjudicated.
  ([#11340 merged implementation](https://github.com/NVIDIA/cccl/pull/11340/files))
- When `insert_or_assign` supports erased slots, search past tombstones for an existing key
  before claiming reusable storage; otherwise assignment can create a duplicate. Retry the
  claim CAS against the erased sentinel actually observed, and recognize a competing insertion
  of the same key as an assignment target. In cooperative code, broadcast the chosen lane's
  outcome, retry the current bucket after a lost claim, and terminate after a complete probe
  cycle. ([#11625 merged implementation and regressions](https://github.com/NVIDIA/cccl/pull/11625))
- Never pass live slot memory to user callbacks by forwarding reference (`for_each` with `auto&&`
  could mutate keys); materialize a copy. Callback wrappers need a non-const `operator()` so
  scalar and CG paths behave alike. (#11095)
- `switch` over an enum: enumerate every case, no `default:`, so new enumerators warn. (#11095)
- Leader election: `invoke_one` / `elect_one` / `ptx::bfind` are preferred but gate on CTK
  (`invoke_one` fixed only on >= 13.2). (#7705)
- Alignment: `cuda::is_aligned`, `cuda::std::assume_aligned`, `reinterpret_cast<uintptr_t>` +
  `% alignof(T) == 0`; never `static_cast` for pointer reinterpretation. Comment intentional
  over-alignment (e.g. `pair<int,int>` to 8 for one 64-bit CAS). (#4898, #6666, #7705)
- Atomics: cudax cuco uses plain CUDA atomics (#10001, merged 2026-11-05) instead of
  `cuda::atomic_ref`. Sub-word CAS operates on the whole aligned 32-bit word, so the word must be
  addressable; there is no 16-byte path yet (tracked in #12389). A lightweight libcu++ facility
  for CUB/Thrust remains the preferred long-term home.

## 8. Types, templates, expressions

- **(recurring)** Reuse CCCL before hand-rolling: `cuda::std::span` (replaced cuco `extent`),
  `dynamic_extent`, `rotl`, `bit_cast`, `as_bytes`, `countl_zero/one`, `to_address`,
  `midpoint`, `binary_search`, `numbers::ln2`, `cuda::bitmask`, `cuda::is_power_of_two`,
  `cuda::ceil_div`, `round_down`, `in_range`, `is_bitwise_comparable`, `__tuple_like`,
  `add_overflow`, `copy_bytes`, `log2`. (#4898, #6666, #7705, #10558, #10740)
- But don't blindly replace specialized code: constant-size `cuda::std::memcmp` lowers to
  byte compares; keep 4/8-byte word-compare specializations via `bit_cast` and document why.
  (#7705)
- **(recurring)** Explicitly sized integers (`::cuda::std::int32_t`, `uint64_t`) used consistently;
  never mix `int` and `int32_t` in one component. Loop counters/small constants `int`/`uint32_t`,
  not `size_t`, except where `sizeof` produces them. Size-like template params are `int`.
  (#4898, #5400, #6666, #8855, #8857, #7705)
- `cuda::std::byte` for raw bytes, not `uint8_t` ("an 8-bit integer, not memory"). (#4898)
- `bit_cast` (not `static_cast`/`reinterpret_cast`) for reinterpretation; assemble 128-bit results
  with `bit_cast<__uint128_t>`; use the global `__uint128_t` builtin (there is no
  `cuda::std::uint128_t`). `bit_cast` is a type conversion, not a load: use it for fixed-size
  keys (`bit_cast` to `array<byte, N>` gives a single wide load) and keep the
  `memcpy`+`assume_aligned` `__load_chunk` path for dynamic byte spans. (#4898, #5400)
- Widen before arithmetic: `size_t{n} * 2` / `static_cast<size_t>(a) * 33 + static_cast<size_t>(b)`,
  never `static_cast<size_t>(n * 2)`; brace-init from the target type instead of casts, and
  avoid chains of nested `static_cast`s. Use `::cuda::std::__num_bits_v<T>` rather than
  `numeric_limits<T>::digits` or `sizeof * CHAR_BIT` locals.
  clang-tidy `bugprone-misplaced-widening-cast` errors under `CCCL_ENABLE_WERROR`. (#7705, #9736,
  #10688/#10695 CI)
- Remove redundant masks/casts (`& 0xFF` on `uint8_t`, `static_cast<T>` of a `T`); uppercase hex
  literals; parentheses in mixed shift/arithmetic; break dense one-liners into named locals.
  (#5400 x4, #6666, #10621, #10740)
- `if constexpr` for compile-time properties (`alignof(T) == 8`), and only `else if constexpr` when
  branches are mutually exclusive. No immediately-invoked lambdas in device code. (#5400, #7705, #8636)
- `_CCCL_TEMPLATE(...) _CCCL_REQUIRES(...)` instead of `enable_if` SFINAE, applied consistently
  once adopted. (#7705, #9720, #9877)
- Namespace-scope constants in headers are `inline constexpr`; class constants `static constexpr`;
  hoist common subexpressions; no derivable members (`__num_buckets` from `__slots.size()`).
  (#6666, #7705)
- State every special member explicitly (defaulted or deleted); don't rely on `= default` ctors of
  small utilities without checking nvcc; single-arg seed ctors `explicit`. (#4898, #6666, #10621)
- Bitwise ops only on unsigned types, enforced by `static_assert`. (#6666, #8855)
- Packed-CAS eligibility checks the slot type's power-of-two size no greater than eight bytes
  plus `cuda::std::is_trivially_copyable_v` in #11340's merged code,
  then sufficient runtime alignment (`atomic_ref<packed>::required_alignment`).
  `cuda::std::pair` lacks cuco::pair's alignment guarantee, so check the storage base and retain
  the fallback. Payload bitwise comparison is a separate requirement. Derive the packed type
  with `__make_nbit_uint_t<__num_bits_v<T>>` and assert supported widths.
  (#7705, #10025; [#11340 review](https://github.com/NVIDIA/cccl/pull/11340#discussion_r4051415732))
- A runtime fast-path decision still instantiates the fallback. Preserve
  `CUDAX_CUCO_DECLARE_BITWISE_COMPARABLE(Type)` for safe explicit payload opt-ins such as `float`;
  invoke it at global scope before the first trait use, on an unqualified trivially copyable
  type. It specializes `cuda::is_bitwise_comparable_v`, so it is broader than one map instance.
  The caller accepts representation equality: padding, NaN payloads and signed zero matter.
  ([#11340 final macro/docs](https://github.com/NVIDIA/cccl/pull/11340/files))
- Guard iterator-trait formation as well as the fast-path body: instantiate
  `iter_reference_t<It>` inside the `if constexpr (contiguous_iterator<It>)` branch, then require
  an actual `value_type&` or `const value_type&` before vectorizing. A conjunction can still form
  an invalid alias for device-only or proxy iterators during host analysis.
  (#11391)
- Initialize test/benchmark members and hash accumulator arrays directly. Preserve xxHash's
  seed/prime values and unsigned arithmetic when replacing element-by-element assignment with
  aggregate initialization; check generated code when initialization touches a hot path.
  (#11629)
- `cuda::is_trivially_copyable_v`, not `cuda::std::…`, as the memcpy precondition (handles `__half2`).
  (#10688)
- Rebuilt refs (`make_copy`, `rebind_*`) thread ALL state (erased-key sentinel, probing scheme via its
  own rebind) and mirror the erased-key dispatch in `fixed_capacity_map::ref()`. (#7705, #10695)
- `const` accessors never cast away const; no reference returns from value accessors;
  `const` on every unmodified local and parameter, consistent const placement per file.
  (#6666, #7705, #10558, #10695)
- Prefer `&&`/`||` to `and`/`or`; `struct` bases need no `public`. (#6666, #7705)
- No CRTP with exactly one implementation; PImpl via `unique_ptr` acceptable for hiding heavy impl
  headers if justified. (#6666, #7705)
- Purge stale comments carried over from cuco that no longer match the code. (#6666)

## 9. API design

- Policy-based customization (`Policy` template param bundling hash + slicing + finalizer, with a
  public `default_*_policy`) over bare functor parameters; `merge` must check policy compatibility.
  (#8857)
- Expose `thread_scope` and `rebind_scope<NewScope>` on every cuco data structure. (#6666, #8857)
- Umbrella `hash<Key, hash_algorithm::…>` selected by enum, one header per algorithm for
  compile-time/binary size; default `xxhash_64`. (#4898, #10621)
- Never drop "unused" overloads without grepping RAPIDS consumers (the byte-span hash overload is
  used by cuDF for strings). (#4898)
- Return types follow the reference implementation: HLL `estimate()` returns `double`; span overload
  of a hasher returns the same width as the single-key overload; span overloads accept
  `span<const Key>`. (#10218, #10621)
- Public device APIs: destination `cuda::std::span` for capacity contract, template arg selects
  thread scope; host bulk APIs come as sync + `_async` pairs. (#10688, #11340)
- Keep host and device `insert_and_find` contracts distinct: host bulk overloads write mapped
  values and insertion flags through output iterators; device overloads return a slot iterator
  and flag. Existing keys return the stored payload with `false`; a full-table miss produces
  the empty-value sentinel / `false` on host and `{end(), false}` on device.
  ([#11340 merged API](https://github.com/NVIDIA/cccl/pull/11340))
- State concurrent-operation and sentinel restrictions explicitly. `insert_and_find` reserves
  the empty mapped-value sentinel for readiness and permits concurrent calls only from its own
  operation family. `insert_or_assign` leaves the winner of concurrent assignments unspecified;
  concurrent lookup and modification are unsupported. Cooperative calls require participation
  by all group members with the same value. Do not promise input-order or last-input-wins
  semantics for duplicate keys. ([#11340](https://github.com/NVIDIA/cccl/pull/11340),
  [#11625](https://github.com/NVIDIA/cccl/pull/11625); merged contracts)
- Container-wide algorithms (`for_each(callback)` over all slots) do not belong on the map; only
  lookup-shaped key-range operations do. (#11095)
- Prefer `cuda::std::equal_to<Key>` as the default comparator; drop scaffolding for removed
  features (`valid_extent`, device `__lower_bound`). (#7705)
- Design questions that block correctness (slot alignment) get a written plan in the PR before code
  lands; deferred work gets a linked issue immediately (e.g. #11334 span-vs-array storage).
  (#10025, #11095)

## 10. Documentation

- Full public docs live on the public class, not internal impl types; document the user-visible
  contract, not the mechanism. Reference types, never `cuda/experimental/__cuco/...` paths.
  (#7705, #10688, #10740)
- Every documented function: `@brief`, `@param[in]/[out]/[in,out]` for EVERY parameter including
  `__stream`, `@return`, `@throws cuda_error …` where applicable, and
  `@note This function synchronizes the given stream. For asynchronous execution use <name>_async.`
  on sync host APIs. Keep `@note` claims accurate (a ctor that only enqueues does not "synchronize").
  (#7705, #9720, #10693, #10694, #10888)
- Document invalidation semantics (rehash invalidates `ref()` and `data()`), side effects of helpers,
  hashing over object representation / non-cryptographic nature, and any intentional accuracy
  degradation (or `static_assert` against it). (#9748, #10621, #10693, #10740)
- `//!` Doxygen style in headers; plain `//` comments in benchmarks; no `// Review:` scratch notes
  inside `//!` blocks; why-comments on non-obvious optimizations so they survive migration.
  (#4898, #5558, #9720, #10740)
- When a default changes (`linear_probing<1>` -> `<4>`), update every mention and flag it breaking;
  update terminology when concepts change (CG -> groups). (#9748, #9877)

## 11. Tests

- **(recurring)** Migrate the FULL upstream cuco test suite for each migrated API (empty, populated,
  discard-output, cleared; key/value, probing, cg-size, bucket-size axes; `bulk_ref_equivalence`,
  `variable_cg`, `layout_equivalence`, `arrow_compat`), using `c2h::type_list` + `C2H_TEST`.
  Migrate bug-fix regression tests along with the fix (signed-min hash, wraparound, prime failure,
  64-bit capacity). (#10521, #10694, #10740, #11449)
- **(recurring)** Negative and boundary cases, not only the happy path: invalid sizes, undersized
  ref storage, empty ranges for both `add` and `contains` (asserting outputs unchanged), duplicates,
  capacity exhaustion, misaligned storage forcing the fallback CAS path, odd sizes (`char3`, 6-byte
  pairs, mixed-width `<uint8_t,uint32_t>`) exercised through insert/contains. (#6666, #10025,
  #10220, #10740, #10757)
- Regression test for every API-contract fix: `static_assert` on return type, const-correctness of
  device merge, source unchanged after merge, callback cannot mutate slots, scalar-path mutable
  callback, repeated cooperative calls in one kernel, both thread scopes and both extents.
  (#10217, #10218, #10688, #11095, #11399)
- Compile-time constraints get compile-fail tests (`cudax_add_cuco_fail_test`) plus positive
  instantiations; drop test configurations the test cannot actually validate. (#10025, #10740)
- Stencil/predicate tests must distinguish `pred(stencil[i])` from `pred(first[i])`. (#10757)
- For `insert_and_find`, check stored values as well as insertion flags for duplicates, unchanged
  outputs on empty ranges, discard iterators, full-table misses, and launch-error propagation.
  Exercise opted-in `float`, larger non-packable pairs, and trivially copyable payloads with
  nontrivial default constructors through scalar/cooperative paths. Test the bitwise opt-in
  trait separately, including cv qualification and a type not opted in. The final #11340 suite
  does not include the misaligned-storage/contention tests removed during review; do not cite
  those intermediate tests as merged coverage.
  ([#11340 merged tests](https://github.com/NVIDIA/cccl/pull/11340/files))
- For launch-error tests, use constant/discard iterators to describe an invalid grid without
  allocating an enormous input. Check both host overloads throw, then verify a valid call still
  succeeds; a deliberate device fault would not test this recoverable failure path.
  ([#11340 launch regression](https://github.com/NVIDIA/cccl/blob/99e70cb54ba20f950b5730a1bb2790dd9f8109f9/cudax/test/cuco/fixed_capacity_map/test_insert_and_find.cu#L274-L311))
- For `insert_or_assign`, test a pre-existing key behind a tombstone (size must not grow), reuse
  of erased storage, competing claims on erased slots, updates in a full table, and termination
  for an absent key in a full table. Cover the scalar/cooperative, static/dynamic extent, and
  device/block atomic-scope branches exercised by the migrated suite.
  ([#11625 merged tests](https://github.com/NVIDIA/cccl/pull/11625/files))
- Test oracles must not reimplement the library function or mirror the header's own narrowing
  rule (that compares the implementation with itself); a few fixed expected values pin a simple
  mapping, with no patterns x types x bounds sweeps and no re-testing of a utility (`cuda::mul_hi`)
  that has its own test. (#12360)
- 128-bit tests guarded by `_CCCL_HAS_INT128()`; expected values from the reference
  implementation; use `_u128` literals from `literal.h`; one `.pass.cpp` per specialization;
  libcu++ `main` runs on both targets, so put bodies in `TEST_FUNC void test()`. Prefer a small
  callable fixture over tables or boolean chains. (#4898, #5400, #9076, #10621, #10791)
- Factor repeated stream + MR setup into a shared helper; use `cuda::launch` in test kernels.
  (#9720, #11399)
- Remove stale test workarounds once the header is fixed (`nv_diag_suppress 20011`). (#11345)
- Grep ALL tests (including sibling components) when renaming a public API; #10989 was a
  follow-up for one missed `cuco::hash`. (#10989)

## 12. Benchmarks

- Benchmarks are library users: no `detail`/`__` APIs, relative helper includes, no Doxygen,
  `cuda::std` integer types. (#5558, #9748, #10740)
- Post before/after comparisons with hardware, CUDA version, workload, hit rate, medians of
  N runs, per-config tables and geomean (e.g. "RTX PRO 6000, CUDA 13.3, 20M int32 keys, 90%
  occupancy, 1.036x geomean"); new fast paths and dropped optimizations need numbers before merge.
  Ping `@NVIDIA/cccl-benchmark-codeowners`. (#9748, #9868, #10621, #11449)
- Keep axes comparable (query range disjoint from build range or fixed hit rate); keep sweeps
  practical (10% occupancy at 1e8 int64 = 16 GiB); avoid materializing outputs not measured;
  name metrics for what they measure (`RelativeError`, not `MeanRelativeError` for one estimate).
  (#5558, #9748, #9889, #10740)
- Keep paper-relevant metrics when porting (L2 hit rate, DRAM throughput) behind an option;
  `message(FATAL_ERROR …)` when CUB is required but disabled; extract shared setup into `common/`.
  (#5558, #9748, #10740)

## 13. CMake, CI, process

- Test registration per directory: `cudax/test/cuco/<component>/CMakeLists.txt` +
  `add_subdirectory`, never in the top-level `cudax/test/CMakeLists.txt`; target names are
  dot-hierarchical (`cuco.fixed_capacity_map.insert_and_contains`). (#9738, #10694/#10695,
  #10740, #10794)
- Check the effective clang-tidy configuration of generated header-test translation units.
  A config beside a header may not apply to a generated `.cpp` in the build tree; do not drop
  existing checks while adding a local override. #11400 landed header cleanups with its proposed
  configuration changes reverted; enabling include-cleaner was deferred.
  ([#11400 review](https://github.com/NVIDIA/cccl/pull/11400#discussion_r4025920699),
  [final disposition](https://github.com/NVIDIA/cccl/pull/11400#issuecomment-5722302337))
- Run locally before pushing: `pre-commit run --all-files` and the ClangCUDA clang-tidy job
  (`./ci/build_tidy.sh -std 17 -arch 75 -cuda clang++`); it caught issues in 3 of 8 migration PRs.
  Report compiler/arch/std/sanitizer results in the PR. (#10688, #10693, #10695, #10757)
- **(recurring)** Sync with upstream cuCollections fixes while migrating (HLL Spark parity #792,
  `register_mask` #779, bloom `contains` layout #842, index-generation fixes). (#6666, #8857,
  #10740, #11449)
- Keep "promote/make public" and rename PRs behaviour-preserving; refactors ride separately.
  Big migrations get a guidelines pass: when a reviewer says "applies throughout", propagate
  everywhere before re-requesting review (several items needed 2-3 rounds). (#4898, #6666, #10621)
- Reply with a reason when declining a suggestion and record offline decisions in the thread;
  turn "follow-up" promises into linked PRs/issues (#11391 cites the #11095 thread). (#6666, #11095)
- PR description: CCCL template's two-item checklist only. Draft PRs need `/ok to test <sha>` from a
  maintainer. Optional RAPIDS-matrix failures are usually upstream skew: confirm and say so, don't
  mask them in cuco code. Flag known conflicts with concurrent PRs on the same files. (#10025, #10688)
