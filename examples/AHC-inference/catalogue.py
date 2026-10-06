"""Lookup helpers for simulations.csv.

The catalogue lists every (nelec, rs, lambda, phase) for which a converged
checkpoint and its Hartree-Fock input file are shipped.
"""
import collections
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'simulations.csv')
# written by run_inference.py so the obs_script/ tools can rebuild the same
# ansatz in a separate process
SELECTION = os.path.join(HERE, 'outputs', 'selection.json')

PHASES = ('ahc', 'liq', 'twc', 'hwc')
MARK = {'ahc': 'A', 'liq': 'L', 'twc': 'T', 'hwc': 'H'}

# `ansatz` distinguishes wavefunctions sharing a phase: 'nn' is the neural
# backflow, 'nn_sbf' the spin dependent (spinor) one. Lowercase marks in the
# grid. `twist` is the twisted boundary condition in fractional reciprocal
# lattice coordinates, blank for the Gamma point.
ANSATZE = ('nn', 'nn_sbf')

# Small nelec=9 system used when run_inference.py is called with no selection,
# so that a referee can check the pipeline runs before committing to a real job.
DUMMY = {
    'nelec': 9,
    'rs': None,
    'lambda': None,
    'phase': 'dummy',
    'ansatz': 'nn',
    'twist': None,
    'checkpoint': os.path.join(HERE, 'dummy_example', 'ckpt_dummy_example.pkl'),
    'h5': os.path.join(HERE, 'dummy_example', 'HFoutput_dummy_example.h5'),
    'etot': None,
    'etot_err': None,
    'note': 'smoke test, not a production run',
}


def load():
    """Return the catalogue as a list of dicts, paths made absolute."""
    with open(CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['nelec'] = int(r['nelec'])
        r['rs'] = float(r['rs'])
        r['lambda'] = float(r['lambda'])
        # both columns are optional, so a hand-edited csv still loads
        r['ansatz'] = r.get('ansatz') or 'nn'
        tw = (r.get('twist') or '').strip()
        r['twist'] = [float(v) for v in tw.split(',')] if tw else None
        r['checkpoint'] = os.path.join(HERE, r['checkpoint'])
        r['h5'] = os.path.join(HERE, r['h5'])
    return rows


def find(nelec, rs, lam, phase, ansatz='nn', twisted=False):
    """Return the matching row, or None.

    `twisted` selects between the Gamma point and the twisted variant of the
    same run; which twist it is, is the catalogue's business, not the caller's.
    """
    for r in load():
        if (r['nelec'] == nelec and r['rs'] == rs
                and r['lambda'] == lam and r['phase'] == phase
                and r['ansatz'] == ansatz and bool(r['twist']) == twisted):
            return r
    return None


def _mark(row):
    """Cell marker for a row: phase letter, prefixed with 't' when twisted."""
    return ('t' if row['twist'] else '') + MARK[row['phase']]


def _grid(sub, rs_all, lam_all):
    """One availability grid: rows rs_all, columns lam_all, markers per cell."""
    cell = collections.defaultdict(set)
    for r in sub:
        cell[(r['rs'], r['lambda'])].add(_mark(r))
    # order by phase, with the untwisted marker before its twisted partner
    order = lambda m: (PHASES.index(_unmark(m)), m.startswith('t'))
    txt = {k: ' '.join(sorted(v, key=order)) for k, v in cell.items()}
    w = max([len(v) + 2 for v in txt.values()] + [8])
    lines = ['   rs\\lam ' + ''.join(f'{l:>{w}g}' for l in lam_all)]
    for rs in rs_all:
        line = f'   {rs:6g} '
        for lam in lam_all:
            line += f'{txt.get((rs, lam), "."):>{w}}'
        lines.append(line)
    return lines


def _unmark(m):
    """Phase name behind a marker, e.g. 'tL' -> 'liq'."""
    letter = m[1:] if m.startswith('t') else m
    return next(p for p, v in MARK.items() if v == letter)


def _legend(sub):
    """Legend line: the crystalline letters, then the liquid and its twisted variant.

    The liquid is the only phase that was ever run with a twist, so the twisted
    marker is spelled out as its own entry rather than as a general `t` prefix.
    """
    crystal = [p for p in PHASES if p != 'liq']
    out = '  ' + '       '.join(f'{MARK[p]}={p}' for p in crystal if any(
        r['phase'] == p for r in sub))
    if any(r['phase'] == 'liq' for r in sub):
        out += f"     {MARK['liq']}=liq, twist=(0,0)"
    tw = next((r['twist'] for r in sub if r['phase'] == 'liq' and r['twist']), None)
    if tw:
        out += f"     t{MARK['liq']} = liq, twist=({tw[0]:g},{tw[1]:g})"
    return [out]


def table(nelec=None):
    """Render the availability grid as a string.

    The spinor-backflow runs get their own grid beside the main one rather than a
    lowercase letter inside it: they exist at only a few points, and packing phase,
    ansatz and boundary condition into one cell made it unreadable. Both grids use
    the same rs rows so they line up; twisted and untwisted runs share a letter.
    """
    rows = [r for r in load() if nelec is None or r['nelec'] == nelec]
    if not rows:
        return f'(no entries for nelec={nelec})'
    out = []
    for ne in sorted({r['nelec'] for r in rows}):
        sub = [r for r in rows if r['nelec'] == ne]
        rs_all = sorted({r['rs'] for r in sub})
        lam_all = sorted({r['lambda'] for r in sub})
        main = [r for r in sub if r['ansatz'] != 'nn_sbf']
        sbf = [r for r in sub if r['ansatz'] == 'nn_sbf']

        out.append(f'  Ne = {ne}')
        out.extend(_legend(sub))
        out.append('')
        left = _grid(main, rs_all, lam_all)
        if not sbf:
            out.extend(left)
        else:
            right = _grid(sbf, rs_all, lam_all)
            head_l = 'neural backflow (default)'
            head_r = 'neural spinor backflow (--ansatz nn_sbf)'
            left = [f'   {head_l}'] + left
            right = [f'   {head_r}'] + right
            # gutter must clear the header too, not just the grid rows
            w = max(len(x) for x in left) + 4
            for a, b in zip(left, right):
                out.append(f'{a:<{w}}{b}')
        # argparse collapses runs of blank lines in the epilog, so the
        # separator between Ne blocks is a line holding a space
        out.extend(['', ' '])
    return '\n'.join(out)


def save_selection(row):
    os.makedirs(os.path.dirname(SELECTION), exist_ok=True)
    with open(SELECTION, 'w') as f:
        json.dump(row, f, indent=2)


def load_selection():
    """Read the selection written by the last run_inference.py invocation."""
    if not os.path.isfile(SELECTION):
        raise SystemExit(
            f'no selection found at {SELECTION}\n'
            'run run_inference.py first, e.g.\n'
            '  python run_inference.py --nelec 36 --rs 18 --lam 0.72 --phase ahc')
    with open(SELECTION) as f:
        return json.load(f)
