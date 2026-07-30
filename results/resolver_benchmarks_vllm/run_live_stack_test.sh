#!/usr/bin/env bash
# Run the LIVE source-resolver stack test inside a real serving-framework ROCm
# container (vLLM or SGLang).
#
# Exercises the three real modules against the actually-installed framework tree
# (no fakes, no GPU -- source scanning only):
#   1. source_env          -> discover_frameworks() dictionaries + metadata
#   2. kernel_source_index -> build + cache the kernel index
#   3. source_resolver_v2  -> resolve real kernels to their file/line
#
# Usage:
#   results/resolver_benchmarks_vllm/run_live_stack_test.sh [tag]
#
# Examples:
#   # vLLM (default)
#   run_live_stack_test.sh v0.26.0
#   # SGLang
#   IMAGE_REPO=lmsysorg/sglang DISCOVER=aiter,sglang EXPECT=aiter,sglang \
#     run_live_stack_test.sh v0.5.5rc0-rocm630
#
# Env:
#   IMAGE_REPO   image repo   (default: vllm/vllm-openai-rocm)
#   DISCOVER     frameworks to scan   -> HYPERLOOM_DISCOVER_ONLY (default: aiter,vllm)
#   EXPECT       frameworks that must be found -> HYPERLOOM_EXPECT_FRAMEWORKS
#                (default: same as DISCOVER)
#   KEEP_IMAGES  keep pulled image afterwards (default: keep; set 0 to remove)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
IMAGE_REPO="${IMAGE_REPO:-vllm/vllm-openai-rocm}"
DISCOVER="${DISCOVER:-aiter,vllm}"
EXPECT="${EXPECT:-$DISCOVER}"
TAG="${1:-v0.26.0}"
IMAGE="$IMAGE_REPO:$TAG"
KEEP_IMAGES="${KEEP_IMAGES:-1}"
TEST_FILE="src/hyperloom/agents/kernel/tests/test_live_source_stack.py"

echo "============================================================"
echo "[$TAG] ensuring image is present: $IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[$TAG] pulling $IMAGE"
  docker pull "$IMAGE" || { echo "[$TAG] PULL FAILED"; exit 1; }
fi

USER_UID="$(id -u)"; USER_GID="$(id -g)"

echo "[$TAG] running live stack test inside container"
docker run --rm -i --entrypoint python3 \
  -v "$REPO_ROOT:$REPO_ROOT" -w "$REPO_ROOT" \
  --user "$USER_UID:$USER_GID" -e HOME=/tmp \
  -e PYTHONPATH="$REPO_ROOT/src" \
  -e HYPERLOOM_LIVE_STACK=1 \
  -e HYPERLOOM_DISCOVER_ONLY="$DISCOVER" \
  -e HYPERLOOM_EXPECT_FRAMEWORKS="$EXPECT" \
  -e HYPERLOOM_KSI_CACHE_DIR="$REPO_ROOT/results/resolver_benchmarks_vllm/ksi_cache" \
  "$IMAGE" "$TEST_FILE"
STATUS=$?

if [ "$KEEP_IMAGES" = "0" ]; then
  echo "[$TAG] removing image to reclaim disk"
  docker rmi "$IMAGE" >/dev/null 2>&1 || true
fi

echo "============================================================"
if [ "$STATUS" -eq 0 ]; then
  echo "[$TAG] LIVE STACK TEST PASSED"
else
  echo "[$TAG] LIVE STACK TEST FAILED (exit $STATUS)"
fi
exit "$STATUS"
