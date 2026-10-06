import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# config_minimal.py (and the deserialize.py it imports) live in obs_script/
sys.path.insert(1, os.path.join(HERE, 'obs_script'))

import catalogue
import fvmc
from config_minimal import get_config
from extract_energy import get_energy

# all run artefacts go here, next to this script regardless of the launch cwd
OUTDIR = os.path.join(HERE, 'outputs')
# the obs_script/ tools default to --fchk checkpoint.pkl in this directory
ACTIVE_CKPT = os.path.join(HERE, 'checkpoint.pkl')


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Run VMC inference from a shipped checkpoint.',
        epilog='available simulations:\n' + catalogue.table(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--nelec', '-N', type=int, default=None,
                   help='number of electrons (default: 16 when selecting a run; '
                        'pass --nelec 36 for the Ne=36 rows)')
    p.add_argument('--rs', type=float, default=None,
                   help='Wigner-Seitz radius')
    p.add_argument('--lam', '--lambda', type=float, default=None, dest='lam',
                   help='dimensionless SOC strength')
    p.add_argument('--phase', default=None, choices=catalogue.PHASES,
                   help='which converged state to load')
    p.add_argument('--ansatz', default='nn', choices=catalogue.ANSATZE,
                   help="wavefunction: 'nn' neural backflow (default), "
                        "'nn_sbf' spin dependent (spinor) backflow")
    p.add_argument('--twisted', action='store_true',
                   help='select the twisted-boundary variant of the run; the '
                        'twist itself is recorded in the catalogue')
    p.add_argument('--iterations', type=int, default=None,
                   help='default: 500 (5 for the dummy example)')
    p.add_argument('--burn_in', type=int, default=None,
                   help='default: 500 (2 for the dummy example)')
    p.add_argument('--dump_every', type=int, default=None,
                   help='trajectory dump frequency in iterations '
                        '(default: config value, 5). Smaller -> more '
                        'snapshots for observables, at no extra sampling cost.')
    return p.parse_args(argv)


def select(args):
    """Resolve the request to a catalogue row; no flags at all -> dummy example."""
    given = {k: getattr(args, k) for k in ('nelec', 'rs', 'lam', 'phase')}
    if all(v is None for v in given.values()):
        return catalogue.DUMMY, True

    missing = [k for k in ('rs', 'lam', 'phase') if given[k] is None]
    if missing:
        sys.exit('missing ' + ', '.join('--' + m for m in missing)
                 + '; pass all of --rs/--lam/--phase, or none for the dummy '
                 'example\n\n' + catalogue.table())

    nelec = 16 if given['nelec'] is None else given['nelec']
    row = catalogue.find(nelec, args.rs, args.lam, args.phase, args.ansatz,
                         args.twisted)
    if row is None:
        sys.exit(f'no run for nelec={nelec} rs={args.rs:g} '
                 f'lambda={args.lam:g} phase={args.phase} '
                 f'ansatz={args.ansatz} '
                 f'{"twisted" if args.twisted else "untwisted"}\n\n'
                 + catalogue.table())
    for key in ('checkpoint', 'h5'):
        if not os.path.isfile(row[key]):
            sys.exit(f'missing file: {row[key]}')
    return row, False


def get_config_params():
    """Ansatz settings for the selected run, as a dict.

    Reads outputs/selection.json so obs_script/runconfig.py rebuilds the same
    ansatz in a separate process.
    """
    row = catalogue.load_selection()
    return dict(
        path=row['h5'],
        # the liquid is translation invariant; every other state breaks it
        full_translation=(row['phase'] == 'liq'),
        wrap_dist_jastrow=True,   # same for every shipped checkpoint
        init_state_orbs=7,        # same for every shipped checkpoint
        # the Ne=16 runs come from an earlier training config whose liquid
        # orbital k-set differs; see config_minimal.CONVENTIONS
        convention='ne16' if row['nelec'] == 16 else 'ne36',
        with_spin_bf=(row['ansatz'] == 'nn_sbf'),
        # fractional reciprocal coordinates; None is the Gamma point
        twist=row['twist'],
    )


def main():
    args = parse_args()
    row, is_dummy = select(args)

    # the dummy example only exists to prove the pipeline runs, so keep it short
    iterations = args.iterations if args.iterations is not None else (5 if is_dummy else 500)
    burn_in = args.burn_in if args.burn_in is not None else (2 if is_dummy else 500)

    os.makedirs(OUTDIR, exist_ok=True)
    catalogue.save_selection(row)
    # the obs_script/ tools read the params from ./checkpoint.pkl
    shutil.copy2(row['checkpoint'], ACTIVE_CKPT)

    what = ('dummy example' if is_dummy else
            f'rs={row["rs"]:g} lambda={row["lambda"]:g} {row["phase"]}'
            + ('' if row['ansatz'] == 'nn' else f' {row["ansatz"]}')
            + ('' if not row['twist'] else
               ' twist=(' + ', '.join(f'{t:g}' for t in row['twist']) + ')'))
    print(f'{what}, nelec={row["nelec"]}, {iterations} iterations, '
          f'{burn_in} burn-in')

    par = get_config_params()
    ckpt_name = os.path.join(OUTDIR, 'ckpt_inference.pkl')

    backflow_dict = {'checkpoint': ACTIVE_CKPT}
    jastrow_dict = {'wrap_dist': par['wrap_dist_jastrow'],
                    'checkpoint': ACTIVE_CKPT}
    orbital_dict = {'checkpoint': ACTIVE_CKPT,
                    'full_translation': par['full_translation'],
                    'init_state': par['init_state_orbs']}

    cfg = get_config(file_name=par['path'], discrete_spin=True,
                     jastrow_params=jastrow_dict, backflow_params=backflow_dict,
                     orbital_params=orbital_dict, save_ckpt_path=ckpt_name,
                     stat_path=os.path.join(OUTDIR, 'data_inference.txt'),
                     out_dir=OUTDIR, convention=par['convention'],
                     with_spin_bf=par['with_spin_bf'],
                     twist_load=par['twist'],
                     mc_chain_checkpoint=None)
    cfg.iterations = iterations
    # not exposed as get_config arguments, so set them on the returned config
    cfg.log.dump_path = os.path.join(OUTDIR, 'trajdump.npy')
    cfg.log.hpar_path = os.path.join(OUTDIR, 'hparams.yaml')
    if args.dump_every is not None:
        cfg.log.dump_every = args.dump_every
    cfg.sample.burn_in = burn_in

    fvmc.infer.main(cfg)

    nelec = -cfg.system.charge
    etot_avg, etot_std, _, _, _, _, _, _ = get_energy(OUTDIR, nelec, nstart=0)
    with open(os.path.join(OUTDIR, 'total_energy.txt'), 'w') as f:
        f.write("etot_avg etot_std\n")
        f.write(f"{etot_avg} {etot_std}\n")
    print(f'Total energy per electron after averaging and unit conversions: {etot_avg} +/- {etot_std} Ry')
    if not is_dummy:
        print(f'Catalogue reference      : {row["etot"]} +/- {row["etot_err"]} Ry')


if __name__ == "__main__":
    main()
