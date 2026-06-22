#!/usr/bin/env bash
set -e
cd "<PROJECT_ROOT>/trimole_hybrid"
<ENV_ROOT>/trimole_bench310/bin/python scripts/strict/update_scoreboard_22.py
