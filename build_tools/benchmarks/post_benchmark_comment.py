#!/usr/bin/env python3
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Posts benchmark results to gist and comments on pull requests.

Requires the environment variables:

- GITHUB_TOKEN: token from GitHub action that has write access on issues. See
  https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token
- GIST_BOT_USER: user name that posts the gist.
- GIST_BOT_TOKEN: token that has write access to gist. Gist will be posted as
  the owner of the token. See
  https://docs.github.com/en/rest/overview/permissions-required-for-fine-grained-personal-access-tokens#gists
"""

import sys
import pathlib

# Add build_tools python dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.with_name("python")))

from typing import Any, Optional
import argparse
import http.client
import json
import os
import requests

from reporting import benchmark_comment

GITHUB_IREE_API_PREFIX = "https://api.github.com/repos/iree-org/iree"
GITHUB_GIST_API = "https://api.github.com/gists"
GITHUB_API_VERSION = "2022-11-28"
MAX_PAGES_TO_SEARCH_PREVIOUS_COMMENT = 10


class APIRequester(object):
  """REST API client that injects proper GitHub authentication headers."""

  def __init__(self, github_token: str):
    self._api_headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {github_token}",
        "X-GitHub-Api-Version": GITHUB_API_VERSION,
    }
    self._session = requests.session()

  def get(self, endpoint: str, payload: Any) -> requests.Response:
    return self._session.get(endpoint,
                             data=json.dumps(payload),
                             headers=self._api_headers)

  def post(self, endpoint: str, payload: Any) -> requests.Response:
    return self._session.post(endpoint,
                              data=json.dumps(payload),
                              headers=self._api_headers)

  def patch(self, endpoint: str, payload: Any) -> requests.Response:
    return self._session.patch(endpoint,
                               data=json.dumps(payload),
                               headers=self._api_headers)


class GithubClient(object):
  """Helper to call Github REST APIs."""

  def __init__(self, requester: APIRequester):
    self._requester = requester

  def post_to_gist(self,
                   filename: str,
                   content: str,
                   verbose: bool = False) -> str:
    """Posts the given content to a new GitHub Gist and returns the URL to it."""

    response = self._requester.post(endpoint=GITHUB_GIST_API,
                                    payload={
                                        "public": True,
                                        "files": {
                                            filename: {
                                                "content": content
                                            }
                                        }
                                    })
    if response.status_code != http.client.CREATED:
      raise RuntimeError(
          f"Failed to create on gist; error code: {response.status_code} - {response.text}"
      )

    response = response.json()
    if verbose:
      print(f"Gist posting response: {response}")

    if response["truncated"]:
      raise RuntimeError(f"Content is too large and was truncated")

    return response["html_url"]

  def get_previous_comment_on_pr(self,
                                 pr_number: int,
                                 gist_bot_user: str,
                                 comment_type_id: str,
                                 verbose: bool = False) -> Optional[int]:
    """Gets the previous comment's id from GitHub."""

    for page in range(1, MAX_PAGES_TO_SEARCH_PREVIOUS_COMMENT + 1):
      response = self._requester.get(
          endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
          payload={
              "per_page": 100,
              "page": page,
              "sort": "updated",
              "direction": "desc"
          })
      if response.status_code != http.client.OK:
        raise RuntimeError(
            f"Failed to get PR comments from GitHub; error code: {response.status_code} - {response.text}"
        )

      comments = response.json()
      if verbose:
        print(f"Previous comment query response on page {page}: {comments}")

      # Find the most recently updated comment that matches.
      for comment in comments:
        if (comment["user"]["login"] == gist_bot_user and
            comment_type_id in comment["body"]):
          return comment["id"]

    return None

  def update_comment_on_pr(self, comment_id: int, content: str):
    """Updates the content of the given comment id."""

    response = self._requester.patch(
        endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/comments/{comment_id}",
        payload={"body": content})
    if response.status_code != http.client.OK:
      raise RuntimeError(
          f"Failed to comment on GitHub; error code: {response.status_code} - {response.text}"
      )

  def create_comment_on_pr(self, pr_number: int, content: str):
    """Posts the given content as comments to the current pull request."""

    response = self._requester.post(
        endpoint=f"{GITHUB_IREE_API_PREFIX}/issues/{pr_number}/comments",
        payload={"body": content})
    if response.status_code != http.client.CREATED:
      raise RuntimeError(
          f"Failed to comment on GitHub; error code: {response.status_code} - {response.text}"
      )


def _parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("comment_json", type=pathlib.Path)
  parser.add_argument("--verbose", action="store_true")
  parser.add_argument("--pr_number", required=True, type=int)
  return parser.parse_args()


def main(args: argparse.Namespace):
  github_token = os.environ.get("GITHUB_TOKEN")
  if github_token is None:
    raise ValueError("GITHUB_TOKEN must be set.")

  gist_bot_user = os.environ.get("GIST_BOT_USER")
  if gist_bot_user is None:
    raise ValueError("GIST_BOT_USER must be set.")

  gist_bot_token = os.environ.get("GIST_BOT_TOKEN")
  if gist_bot_token is None:
    raise ValueError("GIST_BOT_TOKEN must be set.")

  pr_number = args.pr_number
  comment_data = benchmark_comment.CommentData(
      **json.loads(args.comment_json.read_text()))

  gist_client = GithubClient(requester=APIRequester(
      github_token=gist_bot_token))
  gist_url = gist_client.post_to_gist(
      filename=f'iree-full-benchmark-results-{pr_number}.md',
      content=comment_data.full_md,
      verbose=args.verbose)

  pr_client = GithubClient(requester=APIRequester(github_token=github_token))
  previous_comment_id = pr_client.get_previous_comment_on_pr(
      pr_number=pr_number,
      gist_bot_user=gist_bot_user,
      comment_type_id=comment_data.type_id,
      verbose=args.verbose)

  abbr_md = comment_data.abbr_md.replace(
      benchmark_comment.GIST_LINK_PLACEHORDER, gist_url)
  if previous_comment_id is not None:
    pr_client.update_comment_on_pr(comment_id=previous_comment_id,
                                   content=abbr_md)
  else:
    pr_client.create_comment_on_pr(pr_number=pr_number, content=abbr_md)


if __name__ == "__main__":
  main(_parse_arguments())
