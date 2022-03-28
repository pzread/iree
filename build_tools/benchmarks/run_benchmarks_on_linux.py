#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Runs all matched benchmark suites on a Linux device."""

import atexit
import re
import subprocess
import os
import shutil
import sys
import tarfile
import time
from typing import Optional, Sequence, Tuple
from common.benchmark_definition import execute_cmd, execute_cmd_and_get_output, get_benchmark_repetition_count, get_git_commit_hash
from common.benchmark_suite import BenchmarkCase, BenchmarkDriver
from common.common_arguments import build_common_argument_parser
from common.linux_device_utils import get_linux_device_info

LINUX_TMP_DIR = "/tmp/iree-benchmarks-runs"


class LinuxBenchmarkDriver(BenchmarkDriver):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def run_benchmarks_for_category(
      self, benchmark_cases: Sequence[BenchmarkCase]
  ) -> Sequence[Tuple[Optional[str], Optional[str], Optional[Exception]]]:
    results = []
    for benchmark_case in benchmark_cases:
      try:
        (benchmark_filename,
         capture_filename) = self.run_benchmark_and_capture(benchmark_case)
        results.append((benchmark_filename, capture_filename, None))
        # Some grace time.
        time.sleep(1)
      except subprocess.CalledProcessError as e:
        if not self.config.keep_going:
          raise e
        print(f"Processing of benchmark failed with: {e}")
        results.append((None, None, e))

    return results

  def run_benchmark_and_capture(self, benchmark_case: BenchmarkCase):
    flagfile = benchmark_case.flagfile_path
    benchmark_info = benchmark_case.benchmark_info

    bencmark_filename = None
    capture_filename = None

    if not benchmark_case.skip_normal_benchmark:
      cmd = [
          benchmark_case.normal_benchmark_tool_path, f"--flagfile={flagfile}"
      ]
      benchmark_tool = os.path.basename(
          benchmark_case.normal_benchmark_tool_path)
      if benchmark_tool == "iree-benchmark-module":
        cmd.extend([
            "--benchmark_format=json",
            "--benchmark_out_format=json",
            f"--benchmark_out={benchmark_case.benchmark_results_filename}",
        ])
        if self.config.benchmark_min_time:
          cmd.extend([
              f"--benchmark_min_time={self.config.benchmark_min_time}",
          ])
        else:
          repetitions = get_benchmark_repetition_count(benchmark_info.runner)
          cmd.extend([
              f"--benchmark_repetitions={repetitions}",
          ])

      result_json = execute_cmd_and_get_output(
          cmd, verbose=self.verbose, cwd=benchmark_case.benchmark_case_dir)
      if self.verbose:
        print(result_json)

      bencmark_filename = benchmark_case.benchmark_results_filename

    if not benchmark_case.skip_traced_benchmark:
      cmd = [
          "TRACY_NO_EXIT=1", benchmark_case.traced_benchmark_tool_path,
          f"--flagfile={flagfile}"
      ]
      process = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 universal_newlines=True,
                                 shell=True,
                                 cwd=benchmark_case.benchmark_case_dir)
      # Wait for the benchmark result to be available; otherwise will see
      # connection failure when opening the catpure tool.
      while True:
        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
          raise ValueError("Cannot find benchmark result line in the log!")
        if self.verbose:
          print(line.strip())
        if re.match(r"^BM_.+/real_time", line) is not None:
          break

      capture_cmd = [
          self.config.trace_capture_tool, "-f", "-o",
          benchmark_case.capture_filename
      ]
      capture_log = execute_cmd_and_get_output(capture_cmd,
                                               verbose=self.verbose)
      if self.verbose:
        print(capture_log)

      capture_filename = benchmark_case.capture_filename

    print("...benchmark completed")
    return (bencmark_filename, capture_filename)


def main(args):
  device_info = get_linux_device_info(args.device_model, args.verbose)
  if args.verbose:
    print(device_info)

  if args.pin_cpu_freq:
    raise NotImplementedError("Not yet supported pinning CPU frequency.")
  if args.pin_gpu_freq:
    raise NotImplementedError("Not yet supported pinning GPU frequency.")

  benchmark_driver = LinuxBenchmarkDriver.build(
      args=args,
      device_info=device_info,
      git_commit_hash=get_git_commit_hash("HEAD"),
      verbose=args.verbose)

  if not args.no_clean:
    atexit.register(shutil.rmtree, args.tmp_dir)

  benchmark_driver.run()

  benchmark_results = benchmark_driver.get_benchmark_results()
  if args.output is not None:
    with open(args.output, "w") as f:
      f.write(benchmark_results.to_json_str())

  if args.verbose:
    print(benchmark_results.commit)
    print(benchmark_results.benchmarks)

  capture_filenames = benchmark_driver.get_capture_filenames()
  if capture_filenames:
    # Put all captures in a tarball and remove the origial files.
    with tarfile.open(args.capture_tarball, "w:gz") as tar:
      for capture_filename in capture_filenames:
        tar.add(capture_filename)

  errors = benchmark_driver.get_errors()
  if errors:
    print("Benchmarking completed with errors", file=sys.stderr)
    raise RuntimeError(errors)


def parse_argument():
  arg_parser = build_common_argument_parser()
  arg_parser.add_argument("--device_model",
                          default="Unknown",
                          help="Device model")

  return arg_parser.parse_args()


if __name__ == "__main__":
  main(parse_argument())
