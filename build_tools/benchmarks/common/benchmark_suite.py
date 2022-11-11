# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for handling the benchmark suite.

Benchmark artifacts should be generated by building the `iree-benchmark-suites`
CMake target, which put them in the following directory structure:

<root-build-dir>/benchmark_suites
└── <benchmark-category> (e.g., TFLite)
    ├── <benchmark-suite> (e.g., MobileBertSquad-fp32)
    │   ├── <benchmark-case> (e.g., iree-vulkan__GPU-Mali-Valhall__kernel-execution)
    │   │   ├── compilation_statistics.json
    │   │   ├── tool
    │   │   └── flagfile
    │   ├── ...
    │   │   ├── compilation_statistics.json
    │   │   ├── tool
    │   │   └── flagfile
    │   └── <benchmark_case>
    │   │   ├── compilation_statistics.json
    │       ├── tool
    │       └── flagfile
    └── vmfb
        ├── <compiled-iree-model>-<sha1>.vmfb
        ├── ...
        └── <compiled-iree-model>-<sha1>.vmfb
"""

import collections
import os
import pathlib
import re

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from common.benchmark_definition import IREE_DRIVERS_INFOS, DriverInfo
from e2e_test_framework.definitions import iree_definitions
import e2e_test_artifacts.iree_artifacts

# All benchmarks' relative path against root build directory.
BENCHMARK_SUITE_REL_PATH = "benchmark_suites"

MODEL_FLAGFILE_NAME = "flagfile"
MODEL_TOOLFILE_NAME = "tool"


@dataclass
class BenchmarkCase:
  """Represents a benchmark case.

    model_name: the source model, e.g., 'MobileSSD'.
    model_tags: the source model tags, e.g., ['f32'].
    bench_mode: the benchmark mode, e.g., '1-thread,big-core'.
    target_arch: the target CPU/GPU architature, e.g., 'GPU-Adreno'.
    driver_info: the IREE driver configuration.
    benchmark_tool_name: the benchmark tool, e.g., 'iree-benchmark-module'.
    benchmark_case_dir: the path to benchmark case directory.
    run_config: the run config from e2e test framework. This overrides the
      `benchmark_case_dir`.
  """

  model_name: str
  model_tags: Sequence[str]
  bench_mode: Sequence[str]
  target_arch: str
  driver_info: DriverInfo
  benchmark_tool_name: str
  benchmark_case_dir: Optional[str] = None
  run_config: Optional[iree_definitions.E2EModelRunConfig] = None


class BenchmarkSuite(object):
  """Represents the benchmarks in benchmark suite directory."""

  def __init__(self, suite_map: Dict[str, List[BenchmarkCase]]):
    """Construct a benchmark suite.

    Args:
      suites: the map of benchmark cases keyed by category directories.
    """
    self.suite_map = suite_map
    self.category_map = dict((os.path.basename(category_dir), category_dir)
                             for category_dir in self.suite_map.keys())

  def list_categories(self) -> List[Tuple[str, str]]:
    """Returns all categories and their directories.

    Returns:
      A tuple of (category name, category dir).
    """
    category_list = [(name, path) for name, path in self.category_map.items()]
    # Fix the order of category list.
    category_list.sort(key=lambda category: category[0])
    return category_list

  def filter_benchmarks_for_category(
      self,
      category: str,
      available_drivers: Optional[Sequence[str]] = None,
      available_loaders: Optional[Sequence[str]] = None,
      cpu_target_arch_filter: Optional[str] = None,
      gpu_target_arch_filter: Optional[str] = None,
      driver_filter: Optional[str] = None,
      mode_filter: Optional[str] = None,
      model_name_filter: Optional[str] = None) -> Sequence[BenchmarkCase]:
    """Filters benchmarks in a specific category for the given device.
      Args:
        category: the specific benchmark category.
        available_drivers: list of drivers supported by the tools. None means to
          match any driver.
        available_loaders: list of executable loaders supported by the tools.
          None means to match any loader.
        cpu_target_arch_filter: CPU target architecture filter regex.
        gpu_target_arch_filter: GPU target architecture filter regex.
        driver_filter: driver filter regex.
        mode_filter: benchmark mode regex.
        model_name_filter: model name regex.
      Returns:
        A list of matched benchmark cases.
    """,

    category_dir = self.category_map.get(category)
    if category_dir is None:
      return []

    chosen_cases = []
    for benchmark_case in self.suite_map[category_dir]:
      driver_info = benchmark_case.driver_info

      driver_name = driver_info.driver_name
      matched_available_driver = (available_drivers is None or
                                  driver_name in available_drivers)
      matched_drivler_filter = driver_filter is None or re.match(
          driver_filter, driver_name) is not None
      matched_driver = matched_available_driver and matched_drivler_filter

      matched_loader = not driver_info.loader_name or available_loaders is None or (
          driver_info.loader_name in available_loaders)

      target_arch = benchmark_case.target_arch.lower()
      matched_cpu_arch = (cpu_target_arch_filter is not None and re.match(
          cpu_target_arch_filter, target_arch) is not None)
      matched_gpu_arch = (gpu_target_arch_filter is not None and re.match(
          gpu_target_arch_filter, target_arch) is not None)
      matched_arch = (matched_cpu_arch or matched_gpu_arch or
                      (cpu_target_arch_filter is None and
                       gpu_target_arch_filter is None))
      bench_mode = ','.join(benchmark_case.bench_mode)
      matched_mode = (mode_filter is None or
                      re.match(mode_filter, bench_mode) is not None)

      model_name_with_tags = benchmark_case.model_name
      if len(benchmark_case.model_tags) > 0:
        model_name_with_tags += f"-{','.join(benchmark_case.model_tags)}"
      if benchmark_case.run_config is not None:
        # For the new run option, we drop the obscure old semantic and only
        # search on model name and its tags.
        model_and_case_name = model_name_with_tags
      elif benchmark_case.benchmark_case_dir is not None:
        # For backward compatibility, model_name_filter matches against the string:
        #   <model name with tags>/<benchmark case name>
        model_and_case_name = f"{model_name_with_tags}/{os.path.basename(benchmark_case.benchmark_case_dir)}"
      else:
        raise ValueError("Either run_config or benchmark_case_dir must be set.")
      matched_model_name = (model_name_filter is None or re.match(
          model_name_filter, model_and_case_name) is not None)

      if (matched_driver and matched_loader and matched_arch and
          matched_model_name and matched_mode):
        chosen_cases.append(benchmark_case)

    return chosen_cases

  @staticmethod
  def load_from_benchmark_framework(
      run_configs: Sequence[iree_definitions.E2EModelRunConfig]):

    suite_map = collections.defaultdict(list)
    for run_config in run_configs:
      module_gen_config = run_config.module_generation_config
      module_exec_config = run_config.module_execution_config
      target_device_spec = run_config.target_device_spec

      driver_info = None
      for value in IREE_DRIVERS_INFOS.values():
        if value.driver_name != module_exec_config.driver.value:
          continue
        if value.loader_name != "" and value.loader_name != module_exec_config.loader.value:
          continue
        driver_info = value
        break
      if driver_info is None:
        raise ValueError(
            f"Can't map execution config to driver info: {module_exec_config}.")

      arch_info = target_device_spec.architecture.value
      target_arch = f"{arch_info.type}-{arch_info.architecture}-{arch_info.microarchitecture}"

      model = module_gen_config.imported_model.model

      bench_mode = list(module_gen_config.compile_config.tags)
      for tag in module_exec_config.tags:
        if tag not in bench_mode:
          bench_mode.append(tag)

      benchmark_case = BenchmarkCase(
          model_name=model.name,
          model_tags=model.tags,
          bench_mode=bench_mode,
          target_arch=target_arch,
          driver_info=driver_info,
          benchmark_tool_name="iree-benchmark-module",
          run_config=run_config)
      category = model.source_type.value
      suite_map[category].append(benchmark_case)

    return BenchmarkSuite(suite_map=suite_map)

  @staticmethod
  def load_from_benchmark_suite_dir(benchmark_suite_dir: str):
    """Scans and loads the benchmarks under the directory."""

    suite_map: Dict[str, List[BenchmarkCase]] = collections.defaultdict(list)
    for benchmark_case_dir, _, _ in os.walk(benchmark_suite_dir):
      model_dir, benchmark_name = os.path.split(benchmark_case_dir)
      # Take the benchmark directory name and see if it matches the benchmark
      # naming convention:
      #   <iree-driver>__<target-architecture>__<benchmark_mode>
      segments = benchmark_name.split("__")
      if len(segments) != 3 or not segments[0].startswith("iree-"):
        continue

      config, target_arch, bench_mode = segments
      bench_mode = bench_mode.split(",")

      # The path of model_dir is expected to be:
      #   <benchmark_suite_dir>/<category>/<model_name>-<model_tags>
      category_dir, model_name_with_tags = os.path.split(model_dir)
      model_name_parts = model_name_with_tags.split("-", 1)
      model_name = model_name_parts[0]
      if len(model_name_parts) == 2:
        model_tags = model_name_parts[1].split(",")
      else:
        model_tags = []

      with open(os.path.join(benchmark_case_dir, MODEL_TOOLFILE_NAME),
                "r") as f:
        tool_name = f.read().strip()

      suite_map[category_dir].append(
          BenchmarkCase(model_name=model_name,
                        model_tags=model_tags,
                        bench_mode=bench_mode,
                        target_arch=target_arch,
                        driver_info=IREE_DRIVERS_INFOS[config.lower()],
                        benchmark_case_dir=benchmark_case_dir,
                        benchmark_tool_name=tool_name))

    return BenchmarkSuite(suite_map=suite_map)
