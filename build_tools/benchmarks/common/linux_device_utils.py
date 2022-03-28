#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utils for accessing Linux device information."""

import re
from typing import Sequence

from .benchmark_definition import (execute_cmd_and_get_output, DeviceInfo,
                                   PlatformType)


def _get_lscpu_field(field_name: str, verbose: bool = False) -> str:
  output = execute_cmd_and_get_output(["lscpu"], verbose)
  (value,) = re.findall(f"^{field_name}:\s*(.+)", output, re.MULTILINE)
  return value


def get_linux_cpu_arch(verbose: bool = False) -> str:
  return _get_lscpu_field("Architecture", verbose)


def get_linux_cpu_features(verbose: bool = False) -> Sequence[str]:
  return _get_lscpu_field("Flags", verbose).split(" ")


def get_linux_cpu_model(verbose: bool = False) -> str:
  return _get_lscpu_field("Model name", verbose)


def get_linux_device_info(device_model: str = "Unknown",
                          verbose: bool = False) -> DeviceInfo:
  """Returns device info for the Android device.

    Args:
    - device_model: the device model name, e.g., 'ThinkStation P520' 
  """
  return DeviceInfo(
      PlatformType.LINUX,
      # Includes CPU model as it is the key factor of the device performance.
      # Space is not allowed in the model name due to the way we parse it.
      model=f"{device_model}({get_linux_cpu_model(verbose)})".replace(" ", "-"),
      # Currently we only have x86, so CPU ABI = CPU arch.
      cpu_abi=get_linux_cpu_arch(verbose),
      cpu_features=get_linux_cpu_features(verbose),
      # Not yet support GPU benchmark on Linux devices.
      gpu_name="Unknown")
