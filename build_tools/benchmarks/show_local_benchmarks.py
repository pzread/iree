#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Diffs two local benchmark result JSON files.

Example usage:
  python3 diff_local_benchmarks.py --base=/path/to/base_benchmarks.json
                                   --target=/path/to/target_benchmarks.json
"""

import argparse
import os
import collections

from common.benchmark_presentation import *


def get_benchmark_result_markdown(benchmark_files: Sequence[str],
                                  verbose: bool = False) -> str:
  """Gets the full markdown summary of all benchmarks in files."""

  table = collections.defaultdict(list)
  for benchmark_file in benchmark_files:
    benchmarks = aggregate_all_benchmarks([benchmark_file])
    for (name, result) in benchmarks.items():
      table[name].append(result.mean_time)

  return "\n".join(f'"{name}",{",".join(str(time) for time in series)}' for (name, series) in table.items())


def parse_arguments():
  """Parses command-line options."""

  def check_file_path(path):
    if os.path.isfile(path):
      return path
    else:
      raise ValueError(path)

  parser = argparse.ArgumentParser()
  parser.add_argument("files",
                      nargs='+',
                      help="Benchmark results")
  parser.add_argument("--verbose",
                      action="store_true",
                      help="Print internal information during execution")
  args = parser.parse_args()

  return args


if __name__ == "__main__":
  args = parse_arguments()
  print(
      get_benchmark_result_markdown(args.files,
                                    verbose=args.verbose))
