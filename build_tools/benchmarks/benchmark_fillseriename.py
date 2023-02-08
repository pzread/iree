# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import pathlib
import json


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("db_dir", type=pathlib.Path)
  args = parser.parse_args()

  series_db = args.db_dir / "series"

  for series_json in series_db.iterdir():
    series = json.loads(series_json.read_text())
    if "serieName" in series:
      continue
    series["serieName"] = series_json.stem
    series_json.write_text(json.dumps(series))

  series_info_file = args.db_dir / "infos/benchmarks.series.json"
  series_info = json.loads(series_info_file.read_text())
  for key, value in series_info.items():
    if "serieName" in value:
      continue
    value["serieName"] = key
  series_info_file.write_text(json.dumps(series_info))


if __name__ == "__main__":
  main()
