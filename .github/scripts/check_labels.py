#!/usr/bin/env python3
"""Check a PR has required labels."""

from typing import Any

from gitutils import (
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
)
from trymerge import GitHubPR
from github_utils import (
    gh_post_delete_comment,
    gh_post_pr_comment,
)
from label_utils import (
    LABEL_ERR_MSG,
    is_label_err_comment,
    has_required_labels,
)

def delete_all_label_err_comments(pr: "GitHubPR") -> None:
    for comment in pr.get_comments():
        if is_label_err_comment(comment):
            gh_post_delete_comment(pr.org, pr.project, comment.database_id)


def add_label_err_comment(pr: "GitHubPR") -> None:
    # Only make a comment if one doesn't exist already
    if not any(is_label_err_comment(comment) for comment in pr.get_comments()):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Check PR labels")
    parser.add_argument("pr_num", type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    try:
        if not has_required_labels(pr):
            print(LABEL_ERR_MSG)
            add_label_err_comment(pr)
            exit(1)
        else:
            delete_all_label_err_comments(pr)
    except Exception as e:
        pass


if __name__ == "__main__":
    main()
