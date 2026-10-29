"""Read the published task records without modifying their contents."""
import csv
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
def list_tasks():
    with (ROOT/'configs/task_index.csv').open(newline='') as f:
        return list(csv.DictReader(f))
def load_task(task):
    if task not in {p.stem for p in (ROOT/'configs/tasks').glob('*.json')}:
        raise ValueError('Unknown task: '+task)
    return json.loads((ROOT/'configs/tasks'/(task+'.json')).read_text())
def checkpoint_manifest():
    return json.loads((ROOT/'checkpoints/manifest.json').read_text())
