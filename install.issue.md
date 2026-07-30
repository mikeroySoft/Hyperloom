# install.sh 问题报告 —— `ensure_xdit_quality_deps()` 静默把共享的 ROCm PyTorch 换成了 CUDA 版本

**状态：** 已修复（见下方"采用的修复"）。发现于 2026-07-30，在 MI325X pod 上为
`Qwen3-14B-FP8` 启动 `atom` 框架 session 时触发；受影响 pod 已手动恢复（见下方
"已执行的恢复操作"）。

## 位置

`src/hyperloom/inference_optimizer/assets/install.sh`，函数
`ensure_xdit_quality_deps()`（约第 986 行），从主安装流程中无条件调用（约第
1122 行）——**无论 `--framework` 是什么都会执行**，包括 `atom`/`vllm`/`sglang`
这些根本用不到 xDiT 的 session。

```bash
ensure_xdit_quality_deps() {
  log "ensuring xDiT image-quality gate deps (SSIM/LPIPS) in $PYTHON"
  ...
  "$PYTHON" -m pip install --quiet --no-cache-dir \
    "${PIP_EXTRA[@]}" "${missing[@]}" \
    || warn "failed to install xDiT quality deps: ${missing[*]} (gate degrades to MSE-only)"
  ...
}
```

`_XDIT_QUALITY_DEPS=("scikit-image:skimage" "lpips:lpips")`。

## 发生了什么

1. 这台机器上共享的 venv（`/opt/venv`，本 pod 上所有框架 session 共用）原本装的是
   厂商定制的 ROCm 版 PyTorch：`torch==2.10.0+rocm7.2.4.lw.git3d3aa833`（来自 AMD
   私有 wheel 索引 `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/`），
   配套的 `torchvision==0.25.0+rocm7.2.4...` 和
   `torchaudio==2.10.0+rocm7.2.4...` 版本也与之匹配。
2. `ensure_xdit_quality_deps()` 执行了一句裸的
   `pip install scikit-image lpips`——没有指定 `--index-url`，没有加
   `--no-deps`，也没有版本锁定。
3. `lpips` 声明依赖 `torch>=0.4.0`（一个没有上界的宽松约束）。pip 的解析器只对着
   默认 PyPI 索引解析，于是解析到了 PyPI 上自己的 `torch==2.13.0`（一个通用的
   **CUDA** 构建版本），并**升级/替换**了已装的 ROCm 版 torch——同时连带把
   传递依赖里的 `triton` 也从 ROCm 版 `triton==3.6.0+rocm7.2.4...` 换成了
   CUDA 版 `triton==3.7.1`。
4. `pip install` 这条命令本身是 exit 0（成功）。唯一暴露出来的信号是后面一行
   不起眼的 WARN：
   `xDiT quality dep 'lpips' not importable after install (gate excludes it)`
   ——读起来像是"lpips 自己没装上"，完全没有提示"你共享的 PyTorch 刚刚被换成了
   不兼容的版本"。torch/triton 被替换这件事本身是完全静默的。
5. 最终效果：**所有共享这个 venv 的框架/session**（atom、vllm、sglang——这台
   pod 上这几个框架都跑在同一个 `/opt/venv` 里）在下次真正碰 GPU 时都会失效，
   在 `torch.cuda.is_available()` / `torch._C._cuda_init()` 处报错
   `Found no NVIDIA driver`（因为 CUDA-only 的 torch 构建没有 ROCm 后端）。
   这次之所以被发现，是因为（在另一次 commit 里修复了 bug #1 之后）**第二次**
   针对同一个 atom session 重新运行的 install.sh，它自己有一道 torch 完整性
   检查（`torch=... is CUDA-built on a ROCm pod`）在安装阶段就硬性拦截了——
   否则这个问题会在 baseline/profile/kernel_opt 运行过程中静默把这些步骤全部
   打挂。

## 为什么这类 bug 特别危险

- `ensure_xdit_quality_deps()` 对**它自己的安装目标**是 fail-soft 的（`lpips`
  装不上 → 门禁降级为 MSE-only），但对 `pip install` 顺带升级了一个不相关的、
  承重的核心包（`torch`）这件事**完全没有防护**。一次"成功"的 pip 调用
  （exit 0）照样可以把环境搞坏。
- 故障暴露的位置离根因很远：真正的破坏（torch 被换掉）发生在
  `ensure_xdit_quality_deps()` 执行的当下且完全静默，但可见的失败要等到
  很久之后——要么是某个 baseline/profile 任务深处的
  `torch.cuda.is_available() == False`，要么（算运气好）是**下一次**
  install.sh 运行时自己的 ROCm-vs-CUDA 门禁拦下来。
- 影响范围是整台机器级别，不是单个 session 级别：这台 pod 上 atom/vllm/sglang
  全部跑在同一个共享 `/opt/venv` 里，所以一个"这次启动（`--framework atom`）
  根本用不到"的 xDiT-only 安装步骤，把这台机器上所有其他框架的 GPU 访问都
  打挂了。

## 在这台 pod 上执行的恢复操作（手动，不属于 install.sh 的一部分）

```bash
# 从已安装的 dist-info / 未受影响的同批依赖包（torchvision/torchaudio 依然是
# 原来的 ROCm 版本，只有 torch + triton 被换掉）反推出原始 wheel 的准确身份：
#   torch-2.13.0.dist-info 出现在原本 torch-2.10.0+rocm7.2.4.lw.git3d3aa833
#   所在的位置；torchvision==0.25.0+rocm7.2.4.git82df5f59 和
#   torchaudio==2.10.0+rocm7.2.4.git5047768f 是未受影响的同批依赖，
#   可作为版本对照。

/opt/venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/torch-2.10.0%2Brocm7.2.4.lw.git3d3aa833-cp312-cp312-linux_x86_64.whl"

/opt/venv/bin/python -m pip install --no-cache-dir --force-reinstall --no-deps \
  "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/triton-3.6.0%2Brocm7.2.4.git4ed88892-cp312-cp312-linux_x86_64.whl"

# 已验证：
#   torch.__version__          == 2.10.0+rocm7.2.4.git3d3aa833
#   torch.version.hip          == 7.2.53211
#   torch.cuda.is_available()  == True（识别到 AMD Instinct MI325X）
#   pip check                  -> No broken requirements found.
```

保留未处理（不阻塞，纯粹是残留物）：还留有若干孤立的 CUDA-only
`nvidia-cu13-*` / `cuda-toolkit` / `cuda-bindings` 包，是那次 CUDA 版 torch
依赖树的遗留物。`pip check` 显示无冲突，为避免引入不相关的副作用暂未清理。

## 采用的修复（方案三：两者都做）

对 `src/hyperloom/inference_optimizer/assets/install.sh` 的
`ensure_xdit_quality_deps()` 做了两层防御：

1. **`pip install` 加 `--no-deps`**：从源头上阻止 `pip` 在这个可选门禁的安装
   过程中触碰 `torch`/`torchvision`/`scipy` 等的依赖解析。`lpips`/
   `scikit-image` 需要的 `torch`/`torchvision`/`numpy` 等本来就已经在这个
   venv 里（pytorch-xdit 镜像自带），`--no-deps` 只是跳过 pip 对这些已满足
   依赖的重新解析，不影响 `lpips`/`scikit-image` 本身能否 import。已实测
   验证：卸载重装 `lpips --no-deps` 后 torch 版本/build 完全不变，`lpips`
   依然可以正常 import。
2. **安装前后做 torch 完整性校验（纵深防御）**：`pip install` 之前记录
   `torch.__version__`/`torch.version.hip`；只要之前 torch 是可用且是 ROCm
   构建的（`hip` 非空），安装之后立即复查——如果版本变了或 `hip` 变成了空
   （被换成 CUDA build），自动尝试 `pip install --force-reinstall --no-deps
   torch==<之前记录的精确版本>` 回滚；回滚后再次校验，成功则 `warn` 记录
   并继续；如果回滚本身失败，或回滚后依然不是 ROCm 构建，则 `die()` 硬性
   中止安装（而不是像原来那样只 `warn` 后静默继续），避免带着一个被打坏的
   共享 torch 继续跑下去。这一层是防御将来**其他**可选依赖步骤重蹈覆辙，
   `--no-deps` 已经堵住了这次的具体触发路径。

修复已通过 `bash -n` 语法检查，并在受影响 pod 上实测验证 `--no-deps` 行为
符合预期（见上）。

## 同一安装流程中另一个已（单独）修复的 bug

在排查这个问题时，还发现并**修复**了 `src/hyperloom/agents/kernel/scripts/install.sh`
凭证 fallback 逻辑（约第 222 行）里一个独立的 bug（修复已完成，待审阅）：那段
逻辑 source `$REPO_ROOT/.env` 时只 snapshot/restore 了 5 个凭证变量，导致
`USER_DATA_PATH`/`KERNEL_AGENT_ENV`/`MAGPIE_PATH` 等路径变量被**另一个之前跑过
的 session** 遗留在同一个共享 `.env`（由 `upsert_dotenv_var` 持久化写入）里的
值覆盖——只要调用方使用单网关凭证方式（`SAFE_API_KEY`/`OPENAI_BASE_URL`，不设
`ANTHROPIC_*`/`DEEPSEEK_*`）就必然触发这个 fallback。具体修复见
`src/hyperloom/agents/kernel/scripts/install.sh` 的 diff（把 snapshot/restore
范围扩大到该脚本会解析的全部路径变量）。
