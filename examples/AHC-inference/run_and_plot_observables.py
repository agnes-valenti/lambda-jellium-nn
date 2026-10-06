#!/usr/bin/env python3
"""Compute and plot observables from an inference run."""
import os
import subprocess
import sys

HERE  = os.path.dirname(os.path.abspath(__file__))
OBS   = os.path.join(HERE, 'obs_script')
OUT   = os.path.join(HERE, 'outputs')
CACHE = os.path.join(OUT, 'cache-trajdump-obs')
TRAJ  = os.path.join(OUT, 'trajdump.npy')
YML   = os.path.join(OUT, 'hparams.yaml')

observables = [
    "dens",   # charge density
    "gofr",   # g(r)
    "sofk",   # structure factor
    "nofk",   # momentum distribution n(k)
    "ceig",   # chern number (C3 rotation eigenvalues)
]

def run(script, *extra):
    cmd = [sys.executable, os.path.join(OBS, script), TRAJ, '--fyml', YML, *extra]
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)

want = set(observables)

if want & {"dens", "gofr", "sofk"}:
    run('obs.py')
if "ceig" in want:
    run('ceig.py')
if "nofk" in want:
    run('nofk.py')
if "dens" in want:
    run('mdens.py')

# visualize only the computed observables
to_plot = []
if "dens" in want:
    to_plot.append(os.path.join(CACHE, 'mdens'))
if "gofr" in want:
    to_plot += [os.path.join(CACHE, 'gofr'), os.path.join(CACHE, 'vecgofr')]
if "sofk" in want:
    to_plot.append(os.path.join(CACHE, 'sofk'))
if "nofk" in want:
    to_plot.append(os.path.join(CACHE, 'nofk'))
if "ceig" in want:
    to_plot.append(os.path.join(CACHE, 'ceig.npy'))

cmd = [sys.executable, os.path.join(OBS, 'vis_1file.py'), '-s', '-cb', *to_plot]
print('+', ' '.join(cmd), flush=True)
subprocess.run(cmd, check=True, cwd=HERE)
