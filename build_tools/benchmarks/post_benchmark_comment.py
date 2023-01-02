#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Requires

from typing import Any, Optional
import argparse
import json
import markdown_strings as md
import os
import pathlib
import requests

import generate_benchmark_comment as generate_benchmark_comment

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/pzread/iree"
GITHUB_GIST_API_PREFIX = "https://api.github.com/gists"


class APIRequester(object):

  def __init__(self, github_token: str):
    self._api_headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {github_token}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

  def get(self, endpoint: str, payload: Any) -> requests.Response:
    return requests.post(endpoint,
                         data=json.dumps(payload),
                         headers=self._api_headers)

  def post(self, endpoint: str, payload: Any) -> requests.Response:
    return requests.post(endpoint,
                         data=json.dumps(payload),
                         headers=self._api_headers)

  def patch(self, endpoint: str, payload: Any) -> requests.Response:
    return requests.patch(endpoint,
                          data=json.dumps(payload),
                          headers=self._api_headers)


def post_to_gist(requester: APIRequester,
                 filename: str,
                 content: str,
                 verbose: bool = False) -> str:
  """Posts the given content to a new GitHub Gist and returns the URL to it."""
  response = requester.post(endpoint=GITHUB_GIST_API_PREFIX,
                            payload={
                                "public": True,
                                "files": {
                                    filename: {
                                        "content": content
                                    }
                                }
                            })
  if response.status_code != 201:
    raise RuntimeError(
        f"Failed to comment on GitHub; error code: {response.status_code}")

  response = response.json()
  if verbose:
    print(f"Gist posting response: {response}")

  if response["truncated"]:
    raise RuntimeError(f"Content too large and gotten truncated")

  return response["html_url"]


def get_previous_comment_on_pr(requester: APIRequester,
                               pr_number: int,
                               comment_type_id: str,
                               verbose: bool = False) -> Optional[int]:
  """Gets the previous comment's ID from GitHub."""
  # Assume listing 100 comments is enough.
  response = requester.get(
      endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
      payload={"per_page": 100})
  if response.status_code != 200:
    raise RuntimeError(
        f"Failed to get PR comments from GitHub; error code: {response.status_code}"
    )

  response = response.json()
  if verbose:
    print(f"Previous comment query response: {response}")

  # Find the last comment from GITHUB_USER and has the ABBR_PR_COMMENT_TITILE
  # keyword.
  for comment in reversed(response):
    if comment_type_id in comment["body"]:
      return comment["id"]

  return None


def update_comment_on_pr(requester: APIRequester, comment_id: int,
                         content: str):
  """Updates the content of the given comment."""
  response = requester.patch(
      endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/comments/{comment_id}",
      payload={"body": content})
  if response.status_code != 200:
    raise RuntimeError(
        f"Failed to comment on GitHub; error code: {response.status_code}")


def create_comment_on_pr(requester: APIRequester, pr_number: str, content: str):
  """Posts the given content as comments to the current pull request."""
  response = requester.post(
      endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
      payload={"body": content})
  if response.status_code != 201:
    raise RuntimeError(
        f"Failed to comment on GitHub; error code: {response.status_code}")


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("comment_json", type=pathlib.Path)
  parser.add_argument("--verbose", action="store_true")

  return parser.parse_args()


def main(args: argparse.Namespace):
  github_token = os.environ.get("GITHUB_TOKEN")
  if github_token is None:
    raise ValueError("GITHUB_TOKEN must be set.")

  gist_bot_token = os.environ.get("GIST_BOT_TOKEN")
  if gist_bot_token is None:
    raise ValueError("GIST_BOT_TOKEN must be set.")

  comment_data = generate_benchmark_comment.CommentData(
      **json.loads(args.comment_json.read_text()))

  gist_api_requester = APIRequester(github_token=gist_bot_token)
  gist_url = post_to_gist(
      requester=gist_api_requester,
      filename=f'iree-full-benchmark-results-{comment_data.pr_number}.md',
      content=comment_data.full_md,
      verbose=args.verbose)

  pr_api_requester = APIRequester(github_token=github_token)
  previous_comment_id = get_previous_comment_on_pr(
      requester=pr_api_requester,
      pr_number=comment_data.pr_number,
      comment_type_id=comment_data.type_id,
      verbose=args.verbose)
  abbr_md = comment_data.abbr_md.replace(
      generate_benchmark_comment.GIST_LINK_PLACEHORDER, gist_url)
  if previous_comment_id is not None:
    update_comment_on_pr(requester=pr_api_requester,
                         comment_id=previous_comment_id,
                         content=abbr_md)
  else:
    create_comment_on_pr(requester=pr_api_requester,
                         pr_number=args.pr_number,
                         content=abbr_md)


if __name__ == "__main__":
  main(parse_arguments())
