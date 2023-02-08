# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import sys
import pathlib
from typing import Sequence

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

import argparse
from common import benchmark_definition
from e2e_test_framework.models import model_groups
from e2e_test_framework.definitions import common_definitions
from benchmark_suites.iree import benchmark_collections

DEVICE_MAP = {
    "GCP-c2-standard-16 (CPU-x86_64-CascadeLake)":
        benchmark_definition.DeviceInfo(
            platform_type=benchmark_definition.PlatformType.LINUX,
            model="GCP-c2-standard-16",
            cpu_abi="x86_64",
            cpu_uarch="CascadeLake",
            cpu_features=[],
            gpu_name="Unknown"),
    "GCP-a2-highgpu-1g (GPU-NVIDIA-A100-SXM4-40GB)":
        benchmark_definition.DeviceInfo(
            platform_type=benchmark_definition.PlatformType.LINUX,
            model="GCP-a2-highgpu-1g",
            cpu_abi="x86_64",
            cpu_uarch=None,
            cpu_features=[],
            gpu_name="NVIDIA-A100-SXM4-40GB")
}
ARCH_MAP = {
    "CPU-ARM64-v8A":
        common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
    "CPU-RV32-Generic":
        common_definitions.DeviceArchitecture.RV32_GENERIC,
    "CPU-RV64-Generic":
        common_definitions.DeviceArchitecture.RV64_GENERIC,
    "CPU-x86_64-CascadeLake":
        common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
    "GPU-CUDA-SM_80":
        common_definitions.DeviceArchitecture.CUDA_SM80,
    "GPU-Mali-Valhall":
        common_definitions.DeviceArchitecture.MALI_VALHALL,
    "GPU-Adreno":
        common_definitions.DeviceArchitecture.ADRENO_GENERIC,
}


def find_model(model_name: str, model_tags: Sequence[str], model_source: str):
  models = model_groups.ALL
  matched_models = []
  for model in models:
    if (model.name.upper().startswith(model_name.upper()) and
        set(model.tags) >= set(model_tags) and
        model.source_type.value.upper().endswith(model_source.upper())):
      matched_models.append(model)

  return matched_models


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("db_dir", type=pathlib.Path)
  args = parser.parse_args()

  gen_configs, run_configs = benchmark_collections.generate_benchmarks()

  series_db = args.db_dir / "series"
  comments_db = args.db_dir / "comments"

  replacements = {}
  for series_json in series_db.iterdir():
    series_name = series_json.stem
    if "@" not in series_name:
      suffix = series_name.split(" ")[-1]
      comp_info = benchmark_definition.CompilationInfo.from_str(series_name)
      assert series_name == f"{str(comp_info)} {suffix}"
      matched_models = find_model(comp_info.model_name, comp_info.model_tags,
                                  comp_info.model_source)
      if len(matched_models) != 1:
        print(f"{comp_info} {len(matched_models)}")
      matched_model, = matched_models

      matched_gen_configs = []
      for gen_config in gen_configs:
        if benchmark_collections.COMPILE_STATS_TAG not in gen_config.compile_config.tags:
          continue
        if gen_config.imported_model.model != matched_model:
          continue
        assert len(gen_config.compile_config.compile_targets) == 1
        compile_target, = gen_config.compile_config.compile_targets
        if ARCH_MAP[
            comp_info.target_arch] != compile_target.target_architecture:
          continue

        diff = set(gen_config.compile_config.tags).symmetric_difference(
            set(comp_info.compile_tags))
        if "experimental-flags" in diff:
          continue
        if ("kernel-execution" in diff) != ("repeated-kernel" in diff):
          continue

        matched_gen_configs.append(gen_config)

      if len(matched_gen_configs) != 1:
        print(comp_info, matched_gen_configs)
      matched_gen_config, = matched_gen_configs

      model = matched_gen_config.imported_model.model
      compile_config = matched_gen_config.compile_config
      target_archs = []
      for compile_target in compile_config.compile_targets:
        arch = compile_target.target_architecture
        target_archs.append(
            (f"{arch.type.value}-{arch.architecture}-{arch.microarchitecture}-"
             f"{compile_target.target_abi.value}"))
      new_compilation_info = benchmark_definition.CompilationInfo(
          model_name=model.name,
          model_tags=tuple(model.tags),
          model_source=model.source_type.value,
          target_arch=f"[{','.join(target_archs)}]",
          compile_tags=tuple(compile_config.tags))
      replacements[f"{comp_info} {suffix}"] = f"{new_compilation_info} {suffix}"

    else:
      device_name = series_name.split("@")[-1].strip()
      device_info = DEVICE_MAP.get(device_name)
      if device_info is None:
        continue
      bench_info = benchmark_definition.BenchmarkInfo.from_device_info_and_name(
          device_info, series_name)
      assert str(bench_info) == series_name

      matched_models = find_model(bench_info.model_name, bench_info.model_tags,
                                  bench_info.model_source)
      if len(matched_models) != 1:
        print(f"{bench_info} {len(matched_models)}")
      matched_model, = matched_models

      matched_run_configs = []
      for run_config in run_configs:
        if run_config.module_generation_config.imported_model.model != matched_model:
          continue
        if not device_info.model.endswith(
            run_config.target_device_spec.device_name):
          continue

        tags = set(run_config.module_generation_config.compile_config.tags)
        tags.update(run_config.module_execution_config.tags)
        if "experimental-flags" in tags:
          tags -= {"default-flags", "fuse-padding"}

        if tags != set(bench_info.bench_mode):
          continue
        matched_run_configs.append(run_config)

      if len(matched_run_configs) != 1:
        print(bench_info, [(config.module_generation_config.compile_config.id,
                            config.module_execution_config.id)
                           for config in matched_run_configs])
      matched_run_config, = matched_run_configs

      run_tags = matched_run_config.module_execution_config.tags
      compile_tags = matched_run_config.module_generation_config.compile_config.tags
      new_bench_info = benchmark_definition.BenchmarkInfo(
          model_name=matched_model.name,
          model_tags=matched_model.tags,
          model_source=matched_model.source_type.value,
          bench_mode=run_tags,
          compile_tags=compile_tags,
          driver_info=bench_info.driver_info,
          device_info=bench_info.device_info)
      replacements[str(bench_info)] = str(new_bench_info)

  series_info_file = args.db_dir / "infos/benchmarks.series.json"
  series_info = json.loads(series_info_file.read_text())
  new_series_info = dict(series_info)
  for key, value in series_info.items():
    if key.startswith("MiniLML12H384Uncased [int32] (TF)"):
      continue

    try:
      replace = replacements[key]
    except KeyError:
      continue
    del new_series_info[key]
    if replace in new_series_info:
      if value != new_series_info[replace]:
        print(replace)
        print(key)
        print(value)
        print(new_series_info[replace])
    else:
      new_series_info[replace] = value
      orig_data = (series_db / f"{key}.json").read_text()
      (series_db / f"{replace}.json").write_text(orig_data)
      orig_data = (comments_db / f"{key}.json").read_text()
      (comments_db / f"{replace}.json").write_text(orig_data)

  series_info_file.write_text(json.dumps(new_series_info))


if __name__ == "__main__":
  main()
