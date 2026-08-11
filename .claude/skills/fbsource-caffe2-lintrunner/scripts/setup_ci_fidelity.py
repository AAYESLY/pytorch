#!/usr/bin/env python3
"""Set up CLANGTIDY and PYREFLY to run at CI fidelity inside fbsource.

Both linters nominally need a built PyTorch. They do not: everything they
actually consume is either pure-Python codegen or a CMake-configured header
that can be expanded by hand. This script produces, without invoking a
compiler:

  * ``torch/**/*.pyi`` stubs (PYREFLY) via ``tools.pyi.gen_pyi``
  * generated ATen/autograd headers (CLANGTIDY) via ``torchgen.gen`` and
    ``tools/setup_helpers/generate_code.py``
  * the four ``@VAR@`` / ``#cmakedefine`` headers CMake would configure
  * ``build/compile_commands.json`` for the requested translation units
  * ``build/oss.clang-tidy``, plus a ``.lintbin/clang-tidy`` shim that forces
    it (otherwise the Meta-wide fbcode/.clang-tidy is silently inherited)

Run it from fbcode/caffe2 with the lintrunner venv's interpreter.
"""

from __future__ import annotations

import argparse
import json
import re
import stat
import subprocess
import sys
from pathlib import Path


# CMake cache values for a CPU+CUDA configure. None of these gate the code
# paths clang-tidy inspects; they only need to be *defined* so that the
# `#if AT_MKL_ENABLED()`-style tests in the headers compile.
ATEN_CONFIG = {
    "AT_MKLDNN_ENABLED": 0,
    "AT_MKLDNN_ACL_ENABLED": 0,
    "AT_MKL_ENABLED": 0,
    "AT_MKL_SEQUENTIAL": 0,
    "AT_POCKETFFT_ENABLED": 0,
    "AT_NNPACK_ENABLED": 0,
    "CAFFE2_STATIC_LINK_CUDA_INT": 0,
    "AT_BUILD_WITH_BLAS": 1,
    "AT_BUILD_WITH_LAPACK": 1,
    "AT_PARALLEL_OPENMP": 1,
    "AT_PARALLEL_NATIVE": 0,
    "AT_BLAS_F2C": 0,
    "AT_BLAS_USE_CBLAS_DOT": 1,
    "AT_KLEIDIAI_ENABLED": 0,
    "AT_USE_EIGEN_SPARSE": 0,
}

CUDA_CONFIG = {
    "AT_CUDNN_ENABLED": 0,
    "AT_CUSPARSELT_ENABLED": 0,
    "AT_HIPSPARSELT_ENABLED": 0,
    "AT_ROCM_ENABLED": 0,
    "AT_MAGMA_ENABLED": 0,
    "NVCC_FLAGS_EXTRA": "",
}

# Expanded to `#define X 1`; every other #cmakedefine is left undefined.
CMAKE_DEFINED = {"C10_BUILD_SHARED_LIBS", "C10_CUDA_BUILD_SHARED_LIBS"}


def run(argv: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=cwd, check=True)


def version_key(path: Path) -> list[int]:
    return [int(p) for p in re.findall(r"\d+", path.name)]


def find_cuda_include(fbsource: Path) -> Path:
    """Newest third-party CUDA toolkit headers. No toolkit install needed."""
    candidates = sorted(
        (fbsource / "third-party" / "cuda").glob(
            "cuda_*/x64-linux/include_no_implicit"
        ),
        key=lambda p: version_key(p.parent.parent),
    )
    if not candidates:
        raise SystemExit("no third-party/cuda/*/x64-linux/include_no_implicit found")
    return candidates[-1]


def find_fmt_include(fbsource: Path) -> Path:
    """fmt >=10 dropped fmt/core.h, which ATen still includes; pick a version
    that still ships it."""
    candidates = sorted(
        (fbsource / "third-party" / "fmt").glob("*/fmt/include"),
        key=lambda p: version_key(p.parent.parent),
    )
    usable = [c for c in candidates if (c / "fmt" / "core.h").is_file()]
    if not usable:
        raise SystemExit("no third-party/fmt/*/fmt/include containing fmt/core.h")
    return usable[-1]


def expand_at_vars(template: Path, dest: Path, values: dict[str, object]) -> None:
    text = re.sub(r"@(\w+)@", lambda m: str(values[m.group(1)]), template.read_text())
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"  wrote {dest}")


def expand_cmakedefine(template: Path, dest: Path) -> None:
    def sub(m: re.Match[str]) -> str:
        var = m.group(1)
        return f"#define {var} 1" if var in CMAKE_DEFINED else f"/* #undef {var} */"

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(re.sub(r"#cmakedefine (\w+)", sub, template.read_text()))
    print(f"  wrote {dest}")


def gen_python_stubs(py: str, root: Path) -> None:
    """PYREFLY's whole 'needs a built torch' problem is these six files."""
    print("[1/4] generating torch/**/*.pyi stubs (PYREFLY)")
    run(
        [
            py,
            "-m",
            "tools.pyi.gen_pyi",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--deprecated-functions-path",
            "tools/autograd/deprecated.yaml",
            "--out",
            ".",
        ],
        cwd=root,
    )


def gen_cpp_headers(py: str, root: Path) -> None:
    print("[2/4] generating ATen + autograd headers (CLANGTIDY)")
    run(
        [
            py,
            "-m",
            "torchgen.gen",
            "-s",
            "aten/src/ATen",
            "-d",
            "build/aten/src/ATen",
            "--per-operator-headers",
        ],
        cwd=root,
    )
    run(
        [
            py,
            "tools/setup_helpers/generate_code.py",
            "--native-functions-path",
            "aten/src/ATen/native/native_functions.yaml",
            "--tags-path",
            "aten/src/ATen/native/tags.yaml",
            "--gen-lazy-ts-backend",
        ],
        cwd=root,
    )

    print("[3/4] expanding CMake-configured headers")
    build = root / "build"
    expand_at_vars(
        root / "aten/src/ATen/Config.h.in",
        build / "aten/src/ATen/Config.h",
        ATEN_CONFIG,
    )
    expand_at_vars(
        root / "aten/src/ATen/cuda/CUDAConfig.h.in",
        build / "aten/src/ATen/cuda/CUDAConfig.h",
        CUDA_CONFIG,
    )
    expand_cmakedefine(
        root / "torch/headeronly/macros/cmake_macros.h.in",
        build / "torch/headeronly/macros/cmake_macros.h",
    )
    expand_cmakedefine(
        root / "c10/cuda/impl/cuda_cmake_macros.h.in",
        build / "c10/cuda/impl/cuda_cmake_macros.h",
    )


def write_compile_commands(root: Path, fbsource: Path, files: list[str]) -> None:
    py_include = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sysconfig;print(sysconfig.get_paths()['include'])",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    args = [
        "clang++",
        "-std=c++20",  # PyTorch requires C++20; c++17 breaks <compare>
        "-DUSE_CUDA",
        "-DAT_PER_OPERATOR_HEADERS",
        "-DTORCH_CUDA_BUILD_MAIN_LIB",
        f"-I{root}",
        f"-I{root}/aten/src",
        f"-I{root}/build",
        f"-I{root}/build/aten/src",
        f"-I{root}/torch/csrc/api/include",
        f"-I{find_cuda_include(fbsource)}",
        f"-I{find_fmt_include(fbsource)}",
        f"-I{py_include}",
        "-x",
        "c++",
    ]
    build = root / "build"
    build.mkdir(exist_ok=True)
    db = [
        {
            "directory": str(build),
            "arguments": args + [str(root / f)],
            "file": str(root / f),
        }
        for f in files
    ]
    (build / "compile_commands.json").write_text(json.dumps(db, indent=1))
    print(f"  wrote {build / 'compile_commands.json'} ({len(db)} entries)")


def write_oss_clang_tidy(root: Path) -> None:
    """fbcode/caffe2/.clang-tidy sets InheritParentConfig: true. In OSS the
    chain ends there; in fbsource it continues into fbcode/.clang-tidy and
    silently adds the Meta-wide checks (facebook-*, readability-braces-around-
    statements, ...) that the OSS lint job never runs.

    .lintrunner.toml hardcodes --binary=.lintbin/clang-tidy, so the only way to
    force --config-file without editing a tracked file is to shim the binary.
    .lintbin is scratch (deleted on cleanup), so this is idempotent and safe.
    """
    build = root / "build"
    build.mkdir(exist_ok=True)
    config = build / "oss.clang-tidy"
    config.write_text(
        (root / ".clang-tidy")
        .read_text()
        .replace("InheritParentConfig: true", "InheritParentConfig: false")
    )
    print(f"  wrote {config}")

    shim, real = root / ".lintbin/clang-tidy", root / ".lintbin/clang-tidy.real"
    if not shim.is_file():
        print("  .lintbin/clang-tidy missing - run s3_init for clang-tidy first")
        return
    if not real.is_file():
        shim.rename(real)
    shim.write_text(f'#!/bin/bash\nexec {real} --config-file={config} "$@"\n')
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  shimmed {shim} -> {real} with --config-file={config.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--venv-python",
        default=sys.executable,
        help="interpreter with pyyaml + typing_extensions installed",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="caffe2-relative C++ sources to put in compile_commands.json",
    )
    parser.add_argument("--skip-pyi", action="store_true")
    parser.add_argument("--skip-cpp", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    if not (root / ".lintrunner.toml").is_file():
        raise SystemExit("run this from fbcode/caffe2 (no .lintrunner.toml here)")
    fbsource = root.parents[1]

    if not args.skip_pyi:
        gen_python_stubs(args.venv_python, root)
    if not args.skip_cpp:
        gen_cpp_headers(args.venv_python, root)
        print("[4/4] writing compile db + OSS clang-tidy config")
        if args.files:
            write_compile_commands(root, fbsource, args.files)
        write_oss_clang_tidy(root)
    print("done")


if __name__ == "__main__":
    main()
