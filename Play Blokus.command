#!/bin/zsh
# Double-click in Finder to start the local game and open your default browser.
set -e
cd "${0:A:h}"
if [[ ! -x .venv/bin/blokus-rl ]]; then
  print "The project Python environment is missing. See docs/local-play.md."
  read "?Press Return to close."
  exit 1
fi
exec .venv/bin/blokus-rl duo-web --device auto "$@"
