# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
# Tests
################################################################################

iree_py_test(
  NAME
    linux_device_utils_test
  SRCS
    "linux_device_utils_test.py"
)

iree_py_test(
  NAME
    common_arguments_test
  SRCS
    "common_arguments_test.py"
)
