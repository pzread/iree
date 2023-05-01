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
from e2e_test_framework.models import model_groups
from e2e_test_framework.definitions import common_definitions
from benchmark_suites.iree import benchmark_collections

DEVICE_LIST = [
    "Pixel-4 (CPU-ARMv8.2-A)",
    "Pixel-6-Pro (CPU-ARMv8.2-A)",
    "Pixel-6-Pro (GPU-Mali-G78)",
    "XT2201-2 (GPU-Adreno-730)",
]
# ARCH_MAP = {
#     "CPU-ARM64-v8A":
#         common_definitions.DeviceArchitecture.ARMV8_2_A_GENERIC,
#     "CPU-RV32-Generic":
#         common_definitions.DeviceArchitecture.RV32_GENERIC,
#     "CPU-RV64-Generic":
#         common_definitions.DeviceArchitecture.RV64_GENERIC,
#     "CPU-x86_64-CascadeLake":
#         common_definitions.DeviceArchitecture.X86_64_CASCADELAKE,
#     "GPU-CUDA-SM_80":
#         common_definitions.DeviceArchitecture.CUDA_SM80,
#     "GPU-Mali-Valhall":
#         common_definitions.DeviceArchitecture.ARM_VALHALL,
#     "GPU-Adreno":
#         common_definitions.DeviceArchitecture.QUALCOMM_ADRENO,
# }


def find_model(model_name: str, model_tags: Sequence[str], model_source: str):
  models = model_groups.ALL_TFLITE
  matched_models = []
  for model in models:
    if (model.name.upper().startswith(model_name.upper()) and
        set(model.tags) >= set(model_tags) and
        model_source.upper() in model.source_type.value.upper()):
      matched_models.append(model)

  return matched_models


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("db_dir", type=pathlib.Path)
  parser.add_argument("--after", type=int, default=0)
  parser.add_argument("--dry_run", action="store_true")
  args = parser.parse_args()

  gen_configs, run_configs = benchmark_collections.generate_benchmarks()

  series_db = args.db_dir / "series"
  comments_db = args.db_dir / "comments"

  replacements = {}
  for series_json in series_db.iterdir():

    series_data = json.loads(series_json.read_text())
    if int(series_data["lastBuildId"]) < args.after:
      continue

    series_name = series_json.stem
    if "@" not in series_name:
      pass
      # suffix = series_name.split(" ")[-1]
      # comp_info = benchmark_definition.CompilationInfo.from_str(series_name)
      # assert series_name == f"{str(comp_info)} {suffix}"
      # matched_models = find_model(comp_info.model_name, comp_info.model_tags,
      #                             comp_info.model_source)
      # if len(matched_models) != 1:
      #   print(f"{comp_info} {len(matched_models)}")
      # matched_model, = matched_models

      # matched_gen_configs = []
      # for gen_config in gen_configs:
      #   if benchmark_collections.COMPILE_STATS_TAG not in gen_config.compile_config.tags:
      #     continue
      #   if gen_config.imported_model.model != matched_model:
      #     continue
      #   assert len(gen_config.compile_config.compile_targets) == 1
      #   compile_target, = gen_config.compile_config.compile_targets
      #   if ARCH_MAP[
      #       comp_info.target_arch] != compile_target.target_architecture:
      #     continue

      #   diff = set(gen_config.compile_config.tags).symmetric_difference(
      #       set(comp_info.compile_tags))
      #   if "experimental-flags" in diff:
      #     continue
      #   if ("kernel-execution" in diff) != ("repeated-kernel" in diff):
      #     continue

      #   matched_gen_configs.append(gen_config)

      # if len(matched_gen_configs) != 1:
      #   print(comp_info, matched_gen_configs)
      # matched_gen_config, = matched_gen_configs

      # model = matched_gen_config.imported_model.model
      # compile_config = matched_gen_config.compile_config
      # target_archs = []
      # for compile_target in compile_config.compile_targets:
      #   arch = compile_target.target_architecture
      #   target_archs.append(
      #       (f"{arch.type.value}-{arch.architecture}-{arch.microarchitecture}-"
      #        f"{compile_target.target_abi.value}"))
      # new_compilation_info = benchmark_definition.CompilationInfo(
      #     model_name=model.name,
      #     model_tags=tuple(model.tags),
      #     model_source=model.source_type.value,
      #     target_arch=f"[{','.join(target_archs)}]",
      #     compile_tags=tuple(compile_config.tags))
      # metric_id = benchmark_presentation.METRIC_ID_MAP[suffix[1:-1]]
      # replacements[f"{comp_info} {suffix}"] = (
      #     f"{matched_gen_config.composite_id()}-{metric_id}",
      #     f"{new_compilation_info} {suffix}")

    else:
      device_name = series_name.split("@")[-1].strip()
      if device_name not in DEVICE_LIST:
        continue

      (model_name, model_tags, model_source, bench_mode, _,
       backend) = series_name.split(" ")[:6]
      model_tags = model_tags.strip("[]").split(",")
      model_source = model_source.strip("()")
      bench_mode = bench_mode.split(",")

      matched_models = find_model(model_name, model_tags, model_source)
      if len(matched_models) != 1:
        print(f"{series_name} {len(matched_models)}")
      matched_model, = matched_models

      matched_run_configs = []
      for run_config in run_configs:
        if run_config.module_generation_config.imported_model.model != matched_model:
          continue
        device_model = device_name.split(" ")[0]
        if device_model != run_config.target_device_spec.device_name:
          continue
        if "GPU" in device_name and "gpu" not in run_config.target_device_spec.tags:
          continue
        if "CPU" in device_name and "gpu" in run_config.target_device_spec.tags:
          continue
        tmp_bench_mode = set(bench_mode)
        if "little-core" in bench_mode:
          if "little-core" not in run_config.target_device_spec.tags:
            continue
          tmp_bench_mode.remove("little-core")
        if "big-core" in bench_mode:
          if "big-core" not in run_config.target_device_spec.tags:
            continue
          tmp_bench_mode.remove("big-core")

        if "LLVM-CPU" in backend:
          if "vmvx" in str(run_config):
            continue
        if "VMVX" in backend:
          if "vmvx" not in str(run_config):
            continue

        tags = set(run_config.module_generation_config.compile_config.tags)
        tags.update(run_config.module_execution_config.tags)
        tags -= {"system-scheduling", "demote-f32-to-f16"}
        if "experimental-flags" in tags:
          tags -= {"default-flags", "fuse-padding", "mmt4d", "dotprod"}
        if "repeated-kernel" in tags:
          tags -= {"repeated-kernel", "full-inference"}
          tags.add("kernel-execution")

        if tags != tmp_bench_mode:
          continue
        matched_run_configs.append(run_config)

      if len(matched_run_configs) != 1:
        print(series_name, [(config.module_generation_config.compile_config.id,
                             config.module_execution_config.id)
                            for config in matched_run_configs])
      matched_run_config, = matched_run_configs

      print(series_name)
      print(str(matched_run_config))
      replacements[series_name] = (matched_run_config.composite_id,
                                   str(matched_run_config))

  series_info_file = args.db_dir / "infos/benchmarks.series.json"
  series_info = json.loads(series_info_file.read_text())
  new_series_info = dict(series_info)
  unlink_list = []
  for key in series_info.keys():
    try:
      (replace, replace_series_name) = replacements[key]
    except KeyError:
      continue

    value = new_series_info.pop(key)
    value["serieName"] = replace_series_name

    unlink_list.append(key)

    if replace in new_series_info:
      if value == new_series_info[replace]:
        continue
      print(replace)
      print(key)
      print(value)
      print(new_series_info[replace])
      continue

    new_series_info[replace] = value
    series_file = series_db / f"{key}.json"
    orig_data = json.loads(series_file.read_text())
    orig_data["serieName"] = replace_series_name
    if not args.dry_run:
      (series_db / f"{replace}.json").write_text(json.dumps(orig_data))

    comment_file = comments_db / f"{key}.json"
    if comment_file.exists():
      orig_data = comment_file.read_text()
      if not args.dry_run:
        (comments_db / f"{replace}.json").write_text(orig_data)

  if not args.dry_run:
    series_info_file.write_text(json.dumps(new_series_info))

    for key in unlink_list:
      (series_db / f"{key}.json").unlink()
      comment_file = comments_db / f"{key}.json"
      if comment_file.exists():
        comment_file.unlink()


if __name__ == "__main__":
  main()
