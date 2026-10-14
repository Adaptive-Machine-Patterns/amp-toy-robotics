#!/bin/sh
# Fails if anything that should not be published is in the tree:
# machine paths, personal paths, credentials, or files without a licence.
set -e
cd "$(dirname "$0")/.."
bad=$(git ls-files 2>/dev/null || find . -type f -not -path "./.git/*")
if grep -nE "/workspace|/home/|hf_[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH) PRIVATE" $bad \
     --exclude=check_publishable.sh 2>/dev/null; then
  echo "found paths or credentials that must not be published" >&2; exit 1
fi
reuse lint
echo "publishable: ok"
