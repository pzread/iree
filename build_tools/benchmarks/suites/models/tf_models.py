## Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Defines TFLite models."""

from .. import unique_ids
from ..definitions import common_definitions

MINILM_L12_H384_UNCASED_INT32 = common_definitions.Model(
    id=unique_ids.MODEL_MINILM_L12_H384_UNCASED_INT32,
    name="MiniLML12H384Uncased",
    tags=["int32"],
    source_type=common_definitions.ModelSourceType.EXPORTED_TF,
    source_url=
    "https://storage.googleapis.com/iree-model-artifacts/minilm-l12-h384-uncased-tf-model.tar.gz",
    entry_function="predict",
    input_types=["1x512xi32", "1x512xi32", "1x512xi32"])
