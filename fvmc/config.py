from ml_collections import ConfigDict


def default() -> ConfigDict:
    return ConfigDict({

        "seed": 0,

        "verbosity": "WARNING",

        "multi_device": False,

        "quantum_nuclei": False,
        
        "iterations": 10,

        "restart": {
            "params": None,
            "chains": None,
            "states": None,
        },

        "system": system_placeholder(),

        #TODO make a default ansatz parameters
        "ansatz": {},

        "sample": {
            "size": 2048,
            "chains": None,
            "burn_in": 100,
            "sampler": "mcmc",
            "mcmc": {},
            "mixed_mcmc": {},
            "mala": {},
            "mixed_mala": {},
            "hmc": {},
            "adaptive": None,
        },

        "loss": {
            "ke_kwargs": {},
            "pe_kwargs": {},
            "mini_batch": None,
        },

        "log": {
            "stat_path": "data.txt",
            "stat_every": 1,
            "stat_keep": 2, # only back up once, avoid clutter
            "ckpt_path": "checkpoint.pkl",
            "ckpt_every": 100,
            "ckpt_keep": 2, # keep current and last check point
            "dump_path": "trajdump.npy",
            "dump_every": 0, # change to positive number to dump
            "dump_keep": 2, # only back up once, avoid clutter
            "hpar_path": "hparams.yaml",
            "use_tensorboard": False,
            "tracker_path": "tbdata/",
        },

    }, type_safe=False)


def default_tangents() -> ConfigDict:
    return ConfigDict({

        "verbosity": "WARNING",

        "system": system_placeholder(),

        "ansatz": {},

        "sample": {
            "size": 2048,
            "chains": None,
            "burn_in": 100,
            "sampler": "mcmc",
            "mcmc": {},
            "mala": {},
            "hmc": {},
            "adaptive": None,
        },

        "eval_tangents": eval_tangents(),

        "restart": {
            "params": "checkpoint.pkl",
            "chains": None,
            "states": None,
        },

    }, type_safe=False)


def eval_tangents() -> ConfigDict:
    return ConfigDict({
            "ke_kwargs": {"partition_size": None},
            "pe_kwargs": {},
            "reweighting": True,
            "iterations": 100,
            "save_folder": "tangents/",
            "save_mode": "auto",
            "save_every": 1,
        }, type_safe=False)


def system_placeholder() -> ConfigDict:
    return ConfigDict({
        "nuclei": [[0., 0., 0.], [0., 0., 1.]],
        "elems": [1., 1.],
        "charge": 0,
        "spin": None,
        "cell": None,
    }, type_safe=False)

