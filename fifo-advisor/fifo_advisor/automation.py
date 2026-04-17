import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TypeVar

T_unwrap = TypeVar("T_unwrap")


def unwrap(value: T_unwrap | None, error_message: str | None = None) -> T_unwrap:
    if value is None:
        if error_message is None:
            raise ValueError("Unwrapped a None value")
        else:
            raise ValueError(error_message)
    return value


def check_key(key: str | None) -> str:
    if not key:
        raise ValueError("API key not found in .env file")
    else:
        return key


def auto_find_vitis_hls_dir() -> Path | None:
    vitis_hls_bin_path_str = shutil.which("vitis_hls")
    if vitis_hls_bin_path_str is None:
        return None
    vitis_hls_dist_path = Path(vitis_hls_bin_path_str).parent.parent
    return vitis_hls_dist_path


def auto_find_vitis_hls_bin() -> Path | None:
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if vitis_hls_dir is None:
        return None
    vitis_hls_bin = vitis_hls_dir / "bin" / "vitis_hls"
    if not vitis_hls_bin.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but vitis_hls bin not found: {vitis_hls_bin}"
        )
    return vitis_hls_bin


def auto_find_vitis_hls_clang_format() -> Path | None:
    vitis_hls_dist_path = auto_find_vitis_hls_dir()
    if vitis_hls_dist_path is None:
        return None
    vitis_hls_clang_format_path = (
        vitis_hls_dist_path
        / "lnx64"
        / "tools"
        / "clang-3.9-csynth"
        / "bin"
        / "clang-format"
    )
    if not vitis_hls_clang_format_path.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but clang-format not found: {vitis_hls_clang_format_path}"
        )
    return vitis_hls_clang_format_path


def auto_find_vitis_hls_lib_paths() -> list[Path] | None:
    vitis_hls_dir = auto_find_vitis_hls_dir()
    if vitis_hls_dir is None:
        return None

    lib_paths = []

    # /tools/software/xilinx/Vitis_HLS/2024.1/lib/lnx64.o/libxv_hls_llvm3.1.so
    lib_path = vitis_hls_dir / "lib" / "lnx64.o"
    if not lib_path.exists():
        raise RuntimeError(f"Vitis HLS lib path not found: {lib_path}")
    lib_paths.append(lib_path)

    # /tools/xilinx/Vitis_HLS/2024.1/lnx64/lib/csim
    lib_path = vitis_hls_dir / "lnx64" / "lib" / "csim"
    if not lib_path.exists():
        raise RuntimeError(f"Vitis HLS lib path not found: {lib_path}")
    lib_paths.append(lib_path)

    return lib_paths


def get_vitis_hls_include_dir() -> Path | None:
    # vitis_hls_dist_path = get_vitis_hls_dist_path()
    vitis_hls_dist_path = auto_find_vitis_hls_dir()
    if vitis_hls_dist_path is None:
        return None

    vitis_hls_include_dir: Path = vitis_hls_dist_path / "include"
    if not vitis_hls_include_dir.exists():
        raise RuntimeError(
            f"Vitis HLS dir exists but include dir not found: {vitis_hls_include_dir}"
        )
    return vitis_hls_include_dir


class TestCase:
    def __init__(self, dir: Path, name: str):
        self.dir = dir
        self.name = name

        assert self.dir
        # assert self.kernel_fp
        # assert self.kernel_tb_fp
        assert self.hls_script_fp

    @classmethod
    def from_dir(cls, dir: Path, name: str | None = None):
        if name is None:
            name = dir.name
        return cls(dir, name)

    def copy_to(self, dest: Path):
        shutil.copytree(self.dir, dest, dirs_exist_ok=True)
        self.dir = dest

    # @property
    # def src_dir(self):
    #     dir = self.dir / "src"
    #     if not dir.exists():
    #         raise ValueError(f"Expected src dir in {self.dir}, not found")
    #     assert dir.is_dir()
    #     return dir

    # @property
    # def kernel_fp(self):
    #     src_files = list(self.src_dir.glob("*.cpp"))
    #     if len(src_files) != 2:
    #         raise ValueError(
    #             f"Expected 2 cpp files in {self.src_dir}, found {len(src_files)}"
    #         )
    #     matches = [file for file in src_files if not file.name.endswith("_tb.cpp")]
    #     if len(matches) != 1:
    #         raise ValueError(
    #             f"Expected 1 non-testbench cpp file in {self.src_dir}, found {len(matches)}"
    #         )
    #     fp = matches[0]
    #     assert fp.exists()
    #     assert fp.is_file()
    #     return fp

    # @property
    # def kernel_tb_fp(self):
    #     src_files = list(self.src_dir.glob("*.cpp"))
    #     if len(src_files) != 2:
    #         raise ValueError(
    #             f"Expected 2 cpp files in {self.src_dir}, found {len(src_files)}"
    #         )
    #     matches = [file for file in src_files if file.name.endswith("_tb.cpp")]
    #     if len(matches) != 1:
    #         raise ValueError(
    #             f"Expected 1 testbench cpp file in {self.src_dir}, found {len(matches)}"
    #         )
    #     fp = matches[0]
    #     assert fp.exists()
    #     assert fp.is_file()
    #     return fp

    @property
    def hls_script_fp(self):
        fp = self.dir / "hls.tcl"
        if not fp.exists():
            raise ValueError(f"Expected hls.tcl in {self.dir}, not found")
        assert fp.is_file()
        return fp

    @property
    def prj_path(self):
        return self.dir.resolve()

    def build_partial_args(self, log_name: str = "vitis_hls.log") -> list[str]:
        vitis_hls_bin = unwrap(auto_find_vitis_hls_bin())
        args = []
        args.append(str(vitis_hls_bin.resolve()))
        args.append("-l")
        args.append(log_name)
        args.append("hls.tcl")
        args.append(self.name)

        return args

    def build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PRJ_PATH"] = str(self.prj_path.resolve())
        return env

    def run_csim(self):
        args = self.build_partial_args(log_name="vitis_hls__csim.log")
        args.append("csim")
        env = self.build_env()
        p = subprocess.run(
            args, capture_output=True, text=True, bufsize=-1, cwd=self.dir, env=env
        )
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            p.check_returncode()

    def run_synth(self, replace_part: str | None = None):
        args = self.build_partial_args(log_name="vitis_hls__syn.log")
        args.append("syn")
        env = self.build_env()
        p = subprocess.run(
            args, capture_output=True, text=True, bufsize=-1, cwd=self.dir, env=env
        )
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            p.check_returncode()

    def run_cosim(self):
        args = self.build_partial_args(log_name="vitis_hls__cosim.log")
        args.append("cosim")
        env = self.build_env()
        t0 = time.monotonic()
        p = subprocess.run(
            args, capture_output=True, text=True, bufsize=-1, cwd=self.dir, env=env
        )
        t1 = time.monotonic()
        if p.returncode != 0:
            print(p.stdout)
            print(p.stderr)
            p.check_returncode()

        dt = t1 - t0
        cosim_time_fp = self.dir / "cosim_time.txt"
        cosim_time_fp.write_text(f"{dt:.2f}\n")
        print(f"Co-simulation for {self.name} in {self.dir} took {dt:.2f} seconds")

    @property
    def has_solution_dir(self):
        # find the dir prefixed with hls_
        hls_dir_matches = [dir for dir in self.dir.glob("hls_*") if dir.is_dir()]
        if len(hls_dir_matches) == 0:
            return False
        if len(hls_dir_matches) > 1:
            raise ValueError(f"Found more than one hls dir in {self.dir}")

        hls_dir: Path = hls_dir_matches[0]
        solution_dir_path = hls_dir / "solution1"

        if not solution_dir_path.exists():
            return False

        return True

    @property
    def solution_dir(self):
        if not self.has_solution_dir:
            raise ValueError(f"No solution dir found in {self.dir}")
        hls_dir_matches = [dir for dir in self.dir.glob("hls_*") if dir.is_dir()]
        hls_dir: Path = hls_dir_matches[0]
        solution_dir_path = hls_dir / "solution1"
        return solution_dir_path

    @property
    def cosim_dir(self) -> Path:
        raise NotImplementedError
