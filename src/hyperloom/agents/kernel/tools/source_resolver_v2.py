###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Version-robust op -> editable-source resolver (the "finder"), plus a timer.

Instead of trusting the absolute paths captured once in ``op_to_source.json``,
this resolver finds a kernel's source in the *currently installed* framework
tree by its stable identity:

* native kernels: demangle the device symbol -> base name -> look it up in the
  :mod:`kernel_source_index` (self-heals across file moves/renames);
* Triton/Python kernels: resolve the launcher function's ``def`` line via ``ast``.

Every resolve is timed. :func:`latency_report` returns average/percentile
latency keyed by the detected framework versions, so the added cost of the new
logic can be measured directly (the primary ask for this route).

Enabled via ``HYPERLOOM_SOURCE_RESOLVER=v2``; otherwise callers keep the legacy
path. The public :func:`resolve_source` mirrors the legacy signature so it can
be swapped in behind the flag.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import re
import shutil
import subprocess  # nosec B404 - invokes c++filt with a fixed, non-shell argv.
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import kernel_source_index, source_env
from ._bypass_source_resolver import is_editable_source

_OP_TO_SOURCE_JSON = Path(__file__).resolve().parent / "data" / "op_to_source.json"
_PHASE_SUFFIX_RE = re.compile(r"\s*\((?:prefill|decode|prefilldecode|mixed)\)\s*$")

# Kernel kinds we cannot safely edit (assembly labels / CK template instantiations).
_NON_PATCHABLE_KINDS = frozenset({"aiter_asm", "aiter_ck"})

# Kinds whose editable source IS a Python file (the launcher .py is the kernel).
# For every other (native device) kind the op still carries a python launcher
# hint -- the pybind wrapper -- but that is NOT the kernel source, so the
# launcher fallback must not fire for them (it would return a wrong file).
_PYTHON_KERNEL_KINDS = frozenset({"triton", "tilelang"})

# A plain C/C++ identifier (used by the fallback demangler).
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")


@dataclass
class ResolveResult:
    """Outcome of one resolve, including the measured latency."""

    source_file: str
    line: int | None
    symbol: str
    patchable: bool
    method: str
    confidence: str
    elapsed_ms: float
    reason: str = ""

    def as_legacy_tuple(self) -> tuple[str, str]:
        """Legacy ``(source_file, method)`` shape for drop-in compatibility."""
        return (self.source_file, self.method if self.source_file else "unresolved")


# ----------------------------------------------------------------------------
# Latency instrumentation
# ----------------------------------------------------------------------------
@dataclass
class _LatencyBucket:
    version_tag: str
    samples: list[float] = field(default_factory=list)
    index_build_ms: float = 0.0


_LATENCY: dict[str, _LatencyBucket] = {}


def _record_latency(version_tag: str, elapsed_ms: float) -> None:
    bucket = _LATENCY.setdefault(version_tag, _LatencyBucket(version_tag=version_tag))
    bucket.samples.append(elapsed_ms)


def reset_latency() -> None:
    """Clear all recorded latency samples (useful for a fresh benchmark)."""
    _LATENCY.clear()


def latency_report() -> dict[str, Any]:
    """Summarize resolve latency per detected framework version.

    Returns:
        ``{version_tag: {count, avg_ms, p50_ms, p95_ms, max_ms, index_build_ms}}``.
    """
    out: dict[str, Any] = {}
    for tag, bucket in _LATENCY.items():
        s = sorted(bucket.samples)
        n = len(s)
        if n == 0:
            out[tag] = {"count": 0, "index_build_ms": round(bucket.index_build_ms, 2)}
            continue

        def _pct(p: float) -> float:
            idx = min(n - 1, int(round(p * (n - 1))))
            return round(s[idx], 3)

        out[tag] = {
            "count": n,
            "avg_ms": round(sum(s) / n, 3),
            "p50_ms": _pct(0.50),
            "p95_ms": _pct(0.95),
            "max_ms": round(s[-1], 3),
            "index_build_ms": round(bucket.index_build_ms, 2),
        }
    return out


# ----------------------------------------------------------------------------
# Symbol normalization
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=8192)
def _cxxfilt_base(mangled: str) -> str:
    """Demangle via ``c++filt`` when available (``""`` on failure).

    Cached: demangling is pure and the same mangled symbols recur across
    candidates, so we pay the subprocess spawn at most once per symbol.
    """
    if not shutil.which("c++filt"):
        return ""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell.
            ["c++filt", mangled],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _base_from_demangled(name: str) -> str:
    """Extract the base kernel identifier from a demangled/plain symbol."""
    # Keep only the head before params/templates, drop namespaces, then take the
    # last token (drops any leading return type/qualifiers: "void ns::foo" -> "foo").
    head = re.split(r"[(<]", name.strip(), maxsplit=1)[0].split("::")[-1]
    tokens = head.split()
    return tokens[-1] if tokens else ""


def _base_from_mangled(mangled: str) -> str:
    """Fallback: parse Itanium length-prefixed identifiers from a mangled name.

    The length prefix bounds each identifier exactly (e.g. ``18act_and_mul_kernel``
    is precisely 18 chars), so the trailing template marker ``I`` is not glued on.
    """
    names: list[str] = []
    i, n = 0, len(mangled)
    while i < n:
        if mangled[i].isdigit():
            j = i
            while j < n and mangled[j].isdigit():
                j += 1
            length = int(mangled[i:j])
            ident = mangled[j : j + length]
            if _IDENT_RE.match(ident):
                names.append(ident)
            i = j + length
        else:
            i += 1
    if not names:
        return ""
    # Prefer an identifier that looks like a kernel; else the last one.
    for nm in reversed(names):
        if "kernel" in nm.lower():
            return nm
    return names[-1]


@functools.lru_cache(maxsize=8192)
def base_symbol(device_kernel_name: str) -> str:
    """Reduce any device kernel symbol to its stable base name.

    Handles already-demangled names, plain names, and Itanium-mangled names
    (``_Z...``) via ``c++filt`` with a pure-Python fallback.

    Args:
        device_kernel_name: The raw symbol from the trace or JSON key.

    Returns:
        The base kernel identifier (e.g. ``act_and_mul_kernel``), or ``""``.
    """
    raw = (device_kernel_name or "").strip()
    if not raw:
        return ""
    if raw.startswith("_Z"):
        demangled = _cxxfilt_base(raw)
        if demangled and demangled != raw:
            return _base_from_demangled(demangled)
        return _base_from_mangled(raw)
    return _base_from_demangled(raw)


# ----------------------------------------------------------------------------
# Hints (from the existing op_to_source.json, read-only)
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _hints() -> dict[str, Any]:
    """Load ``op_to_source.json`` as a hints source (``{}`` on failure)."""
    try:
        with open(_OP_TO_SOURCE_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _containers(framework: str) -> tuple[str, ...]:
    """Container search order: the hinted framework first, else sglang then vllm."""
    return (framework,) if framework in ("vllm", "sglang") else ("sglang", "vllm")


def _hint_kind_patchable(op_name: str, framework: str) -> tuple[str, bool]:
    """Cheap gate lookup: (kernel_kind, patchable) with NO demangling.

    Kept separate from :func:`_hint_bases` so the non-patchable gate in
    :func:`resolve` can bail before paying for any ``c++filt`` demangling.
    """
    entry = _hints().get(op_name)
    if not isinstance(entry, dict):
        return "", True
    patchable = bool(entry.get("patchable", True))
    for cont in _containers(framework):
        for leaf in (entry.get(cont) or {}).values():
            if isinstance(leaf, dict) and leaf.get("kernel_kind"):
                return str(leaf["kernel_kind"]), patchable
    return "", patchable


def _hint_bases(op_name: str, framework: str) -> list[str]:
    """Demangle the entry's kernel names to base symbols (called lazily)."""
    entry = _hints().get(op_name)
    if not isinstance(entry, dict):
        return []
    bases: list[str] = []
    for cont in _containers(framework):
        for kname in (entry.get(cont) or {}):
            b = base_symbol(kname)
            if b and b not in bases:
                bases.append(b)
    return bases


def _hint_launcher(op_name: str) -> tuple[str, str]:
    """From hints: (relative_launcher_path, func_name) or ``("", "")``."""
    entry = _hints().get(op_name)
    if not isinstance(entry, dict):
        return "", ""
    launchers = entry.get("python_launcher_path") or []
    if not isinstance(launchers, list) or not launchers:
        return "", ""
    m = re.match(r"\s*([^()]+)\((\d+)\):\s*(\S+)", str(launchers[0]))
    if not m:
        return "", ""
    return m.group(1).strip(), m.group(3).strip()


# ----------------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------------
def _rank_records(records: list[dict[str, object]], framework: str) -> list[dict[str, object]]:
    """Rank candidate definition records: framework hint > arch tag > path len."""
    arch = os.environ.get("HYPERLOOM_TARGET_ARCH", "").strip().lower()
    fw = (framework or "").lower()

    def score(rec: dict[str, object]) -> tuple[int, int, int]:
        path = str(rec.get("file", "")).lower()
        fw_match = 1 if fw and rec.get("framework") == fw else 0
        arch_match = 1 if arch and arch in path else 0
        # Prefer shorter paths (canonical location over vendored copies).
        return (fw_match, arch_match, -len(path))

    return sorted(records, key=score, reverse=True)


def _verify_symbol(path: str, base: str) -> bool:
    """Confirm ``base`` actually appears in ``path`` (guards stale index)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return base in text


@functools.lru_cache(maxsize=2048)
def _launcher_line(abs_path: str, func: str) -> int | None:
    """Return the ``def`` line of ``func`` in a Python file via ``ast``.

    Cached by (path, func): a launcher file is stable within a run, so repeated
    ops sharing the same launcher parse it only once.
    """
    try:
        tree = ast.parse(Path(abs_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return node.lineno
    return None


def _resolve_launcher(op_name: str, framework: str) -> tuple[str, int | None]:
    """Lazily resolve a Python launcher to (abs_file, def_line)."""
    rel, func = _hint_launcher(op_name)
    if not rel or not func:
        return "", None
    fw = source_env.discover_frameworks()
    pkg = rel.split("/", 1)[0]
    fr = fw.get(pkg)
    bases = [fr.root.parent] if fr else []
    bases += [f.root.parent for f in fw.values()]
    for base_dir in bases:
        cand = base_dir / rel
        if cand.is_file():
            return str(cand), _launcher_line(str(cand), func)
    return "", None


def resolve(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
    index: kernel_source_index.SourceIndex | None = None,
) -> ResolveResult:
    """Resolve an op/kernel to its editable source in the installed tree (timed).

    Args:
        op_name: Launching op name (e.g. ``_C::silu_and_mul``).
        framework: Serving framework hint (``vllm``/``sglang``).
        device_kernel_name: Device kernel symbol from the trace (authoritative).
        index: Optional prebuilt index (built/cached if omitted).

    Returns:
        A :class:`ResolveResult` with the live file/line, patchability, method,
        and the measured ``elapsed_ms``.
    """
    started = time.perf_counter()
    idx = index if index is not None else kernel_source_index.load_or_build()

    def finish(res: ResolveResult) -> ResolveResult:
        res.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        _record_latency(idx.version_tag, res.elapsed_ms)
        return res

    key = _PHASE_SUFFIX_RE.sub("", op_name or "")
    # Cheap gate first (no demangling): ASM / CK templates and explicitly
    # non-patchable ops bail before we spend any c++filt spawns.
    hint_kind, hint_patchable = _hint_kind_patchable(key, framework)
    if hint_kind in _NON_PATCHABLE_KINDS or not hint_patchable:
        return finish(
            ResolveResult("", None, "", False, "non_patchable", "hint", 0.0, reason=hint_kind or "not patchable")
        )

    # 1) Symbol-first (authoritative): the trace's device kernel name.
    candidate_bases: list[str] = []
    if device_kernel_name:
        b = base_symbol(device_kernel_name)
        if b:
            candidate_bases.append(b)
    # 2) Fallback to hint-derived base names (demangled lazily, only now).
    for b in _hint_bases(key, framework):
        if b not in candidate_bases:
            candidate_bases.append(b)

    for base in candidate_bases:
        records = _rank_records(idx.lookup(base), framework)
        for rec in records:
            path = str(rec.get("file", ""))
            if not is_editable_source(path):
                continue
            if not _verify_symbol(path, base):
                continue
            return finish(
                ResolveResult(path, int(rec.get("line") or 0) or None, base, True, "symbol_index", "high", 0.0)
            )

    # 3) Triton/TileLang launcher: the .py IS the editable kernel source. Native
    # device kernels also carry a launcher hint (their pybind wrapper) that is NOT
    # their source, so we skip this for symbols that look native (mangled ``_Z``
    # or a C++ signature ``::``/``<``) and key off the concrete symbol, not the
    # op-level kind hint (one op can mix a triton kernel with native ones).
    dkn = device_kernel_name or ""
    looks_native = dkn.startswith("_Z") or "::" in dkn or "<" in dkn
    if not looks_native and (hint_kind in _PYTHON_KERNEL_KINDS or not candidate_bases):
        launcher_path, launcher_line = _resolve_launcher(key, framework)
        if launcher_path and is_editable_source(launcher_path):
            return finish(
                ResolveResult(launcher_path, launcher_line, "", True, "launcher_ast", "medium", 0.0)
            )

    return finish(ResolveResult("", None, "", False, "unresolved", "none", 0.0, reason="no live match"))


def resolve_source(
    op_name: str,
    *,
    framework: str = "",
    device_kernel_name: str = "",
) -> tuple[str, str]:
    """Legacy-compatible wrapper: returns ``(source_file, method)``.

    Drop-in for ``_bypass_source_resolver.resolve_source`` when
    ``HYPERLOOM_SOURCE_RESOLVER=v2``.
    """
    return resolve(op_name, framework=framework, device_kernel_name=device_kernel_name).as_legacy_tuple()


def is_enabled() -> bool:
    """Whether the v2 resolver is selected via ``HYPERLOOM_SOURCE_RESOLVER``."""
    return os.environ.get("HYPERLOOM_SOURCE_RESOLVER", "").strip().lower() == "v2"


# ----------------------------------------------------------------------------
# Latency benchmark CLI
# ----------------------------------------------------------------------------
def _sample_candidates(top_k: int) -> list[dict[str, str]]:
    """Build sample candidates from op_to_source.json keys (op + one kernel)."""
    out: list[dict[str, str]] = []
    for op_name, entry in _hints().items():
        if not isinstance(entry, dict):
            continue
        kname = ""
        for cont in ("vllm", "sglang"):
            keys = list((entry.get(cont) or {}).keys())
            if keys:
                kname = keys[0]
                break
        out.append({"op_name": op_name, "device_kernel_name": kname})
        if top_k and len(out) >= top_k:
            break
    return out


def _main(argv: list[str] | None = None) -> int:
    """CLI: ``--bench`` times the finder over sample kernel candidates."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark the v2 source finder latency.")
    parser.add_argument("--bench", action="store_true", help="Run the latency benchmark.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of candidates to resolve.")
    parser.add_argument("--framework", default="vllm", help="Framework hint (vllm/sglang).")
    parser.add_argument("--candidates", default="", help="Optional JSON file: [{op_name, device_kernel_name}].")
    args = parser.parse_args(argv)

    reset_latency()
    fw = source_env.discover_frameworks()
    if not fw:
        print("No frameworks (vllm/sglang/aiter) discovered; cannot benchmark.")
        return 1

    t0 = time.perf_counter()
    index = kernel_source_index.build_index(fw)
    index_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    _LATENCY.setdefault(index.version_tag, _LatencyBucket(version_tag=index.version_tag)).index_build_ms = index_ms

    if args.candidates:
        try:
            cands = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Failed to read candidates: {exc}")
            return 1
    else:
        cands = _sample_candidates(args.top_k)

    resolved = 0
    for c in cands:
        res = resolve(
            c.get("op_name", ""),
            framework=args.framework,
            device_kernel_name=c.get("device_kernel_name", ""),
            index=index,
        )
        if res.source_file:
            resolved += 1

    report = latency_report()
    print(f"Version: {index.version_tag}")
    print(f"Index build: {index_ms} ms ({index.symbol_count} symbols / {index.file_count} files)")
    print(f"Candidates: {len(cands)} | resolved: {resolved}")
    for tag, stats in report.items():
        print(f"Latency[{tag}]: {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ResolveResult",
    "resolve",
    "resolve_source",
    "base_symbol",
    "is_enabled",
    "latency_report",
    "reset_latency",
]
