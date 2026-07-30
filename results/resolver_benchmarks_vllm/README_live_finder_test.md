# Live Op-Source Finder — test guide

A "live finder" that maps a running GPU kernel back to its **editable source
file + line** by scanning the *actually-installed* framework code at runtime,
instead of trusting a stale precomputed JSON.

## The 6 files

| File | Role (one line) |
|---|---|
| `src/hyperloom/agents/kernel/tools/source_env.py` | Find installed libraries (vLLM/SGLang/aiter), versions, and their `csrc` folders; make a version `fingerprint`. |
| `src/hyperloom/agents/kernel/tools/kernel_source_index.py` | Scan native sources for `__global__` kernels → `{file, line}`; cache it, keyed by the fingerprint. |
| `src/hyperloom/agents/kernel/tools/source_resolver_v2.py` | Look up one kernel: native → index (`symbol_index`); Triton/TileLang → `.py` via `ast` (`launcher_ast`); else `unresolved`. |
| `src/hyperloom/agents/kernel/tools/_bypass_source_resolver.py` | Helper `is_editable_source()`: is a path patchable (native `.cu/...` or repo `.py`)? Rejects `/tmp`/generated. |
| `src/hyperloom/agents/kernel/tests/test_live_source_stack.py` | Real end-to-end test that runs all three modules against a live container. |
| `results/resolver_benchmarks_vllm/run_live_stack_test.sh` | Launcher: runs the test inside a real vLLM/SGLang ROCm container. |

## How they work together
- `source_env` finds the libraries →
- `kernel_source_index` builds/caches the kernel "phone book" →
- `source_resolver_v2` answers "kernel X lives here" using that phone book
  (native) or `ast` (Python kernels).
- The test drives this whole chain on a real image; the shell script just
  launches it. **No GPU is needed** (it only reads source files).

## How to run

vLLM image (default):

```bash
results/resolver_benchmarks_vllm/run_live_stack_test.sh v0.26.0
```

SGLang image (set repo + expected frameworks):

```bash
IMAGE_REPO=lmsysorg/sglang DISCOVER=aiter,sglang EXPECT=aiter,sglang \
  results/resolver_benchmarks_vllm/run_live_stack_test.sh v0.5.16-rocm700-mi35x
```

The image is pulled automatically if missing.

Env knobs: `IMAGE_REPO` (image repo), `DISCOVER` (frameworks to scan),
`EXPECT` (frameworks that must be found), `KEEP_IMAGES=0` (delete image after).

## How to read the output (6 sections)
- **source_env** — discovered libraries, versions, `csrc` folders, `fingerprint`, and 10 real kernels found in source.
- **kernel_source_index** — `N symbols / M files`, 10 records verified at real `file:line`, and cache write + reload (hit).
- **source_resolver_v2 (native)** — 10 native kernels resolved to `file.cu:line`.
- **full cache** — writes the complete index to `ksi_cache/` (machine JSON + a readable JSON).
- **resolve real .py launchers** — real Triton/TileLang ops resolved to their `.py` (and `def` line when available).
- **real Triton .py kernels** — real `@triton.jit` kernels found and confirmed editable.
- Final line: `6/6 sections passed` → `LIVE STACK TEST PASSED`.

## How to analyze the results
- **All pass** → the finder correctly discovers, indexes, and resolves native + Python kernels for that image/version.
- **Inspect by hand** in `results/resolver_benchmarks_vllm/ksi_cache/`:
  - `ksi_<fingerprint>.json` — the raw cache the tool uses.
  - `ksi_<fingerprint>_readable.json` — sorted `kernel → [file:line]` map.
- **On failure** the traceback names the section + the offending kernel/op, so you can open that `file:line` directly.
- **Across versions** a different `fingerprint`/`version_tag` means the code changed; the index rebuilds automatically, so counts can legitimately differ per image.

## Key libraries & terms (plain language)
- **`ast` (Abstract Syntax Tree)** — Python's built-in parser that turns source
  into a tree of code elements. We use it to find a function's exact `def` line
  in a `.py` kernel **without importing or running it** (safe and fast).
- **`importlib` (`find_spec`)** — asks "where is package X installed?" without
  importing it. Locates vLLM/SGLang/aiter on disk.
- **`os.walk`** — recursively lists files in a folder tree; used to scan sources.
- **`dataclasses`** — shorthand for simple record classes (`FrameworkRoot`,
  `SourceIndex`, `ResolveResult`).
- **`hashlib`** — builds the short `fingerprint` (cache key) from library paths +
  versions + folder timestamps.
- **`functools.lru_cache`** — remembers results so the same lookup (e.g. one
  `.py` launcher) is parsed only once.
- **`subprocess` + `c++filt`** — `c++filt` converts mangled C++ symbols
  (`_ZN4vllm...`) back to readable names; called via `subprocess`, with a
  pure-Python fallback if missing.
- **`json`** — reads op hints (`op_to_source.json`) and the index cache.
- **Terms**: *demangle* = mangled C++ name → readable base name;
  *fingerprint* = version-aware cache key; *symbol_index* = native-kernel lookup;
  *launcher_ast* = Python-kernel lookup.
```
