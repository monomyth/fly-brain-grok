# shellcheck disable=SC2034
# Paths for ~/robotics/grok on isengard. Source from the wrappers.

GROK_ROOT="${GROK_ROOT:-$HOME/robotics/grok}"
FB="${FB:-$HOME/fly-brain-grok}"
FB_PY="${FB_PY:-$FB/.venv/bin/python}"
ORBBEC_PY="${ORBBEC_PY:-$HOME/robotics/orbbec/.venv/bin/python}"
export PYTHONPATH="${FB}/controller"
export FLYBRAIN_DATA="${FB}/data"
OUT="${GROK_ROOT}/out"
mkdir -p "$OUT" "$OUT/frames"
