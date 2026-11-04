#!/usr/bin/env bash
# Cut a release: bump the version, test, commit, tag vX.Y.Z on main and push both.
# Installed apps update to the newest release tag on their next launch.
#
#   tools/release.sh 2.2.0
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: tools/release.sh X.Y.Z  (current: $(sed -n 's/^__version__ = "\(.*\)"/\1/p' tvshow_noisemaker/__init__.py))" >&2
    exit 2
fi
TAG="v$VERSION"

[[ "$(git rev-parse --abbrev-ref HEAD)" == "main" ]] || { echo "release from main" >&2; exit 1; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "commit or stash your changes first" >&2; exit 1; }
git fetch --quiet --tags origin main
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || { echo "main is not in sync with origin/main" >&2; exit 1; }
! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "$TAG already exists" >&2; exit 1; }

PY="${PYTHON:-$( [[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3 )}"
"$PY" -m pytest -q

PREV="$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || true)"
NOTES="$(git log --format='- %s' ${PREV:+"$PREV"..HEAD})"

sed -i.bak "s/^__version__ = \".*\"/__version__ = \"$VERSION\"/" tvshow_noisemaker/__init__.py
sed -i.bak "s/^Version: .*/Version: $VERSION/" README.md
rm -f tvshow_noisemaker/__init__.py.bak README.md.bak
git add tvshow_noisemaker/__init__.py README.md
git commit -qm "release: $TAG"
git tag -a "$TAG" -m "$TAG" -m "$NOTES"
git push --quiet origin main "$TAG"
echo "Released $TAG"
