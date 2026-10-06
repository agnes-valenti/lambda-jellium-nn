import os
#import sys
#sys.path.append('../../../../')

import sys

from config_minimal import get_config
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from run_inference import get_config_params, OUTDIR, ACTIVE_CKPT

def get_runconfig(nelec, rs):
  par = get_config_params()
  
  checkpoint_jastrow_current = ACTIVE_CKPT
  checkpoint_orbs_current = ACTIVE_CKPT
  checkpoint_backflow_current = ACTIVE_CKPT
  mc_chain_checkpoint_current = None

  ckpt_name = os.path.join(OUTDIR, 'ckpt_save.pkl')
  
  backflow_dict = {}
  backflow_dict['checkpoint'] = checkpoint_backflow_current

  jastrow_dict = {}
  jastrow_dict['wrap_dist'] = par['wrap_dist_jastrow']
  jastrow_dict['checkpoint'] = checkpoint_jastrow_current

  orbital_dict = {}
  orbital_dict['checkpoint'] = checkpoint_orbs_current
  orbital_dict['full_translation'] = par['full_translation']
  orbital_dict['init_state'] = par['init_state_orbs']


    
    
  return get_config(file_name=par['path'], discrete_spin=True,
                    jastrow_params=jastrow_dict, orbital_params=orbital_dict,
                    backflow_params=backflow_dict, save_ckpt_path=ckpt_name,
                    out_dir=OUTDIR, convention=par['convention'],
                    with_spin_bf=par['with_spin_bf'],
                    twist_load=par['twist'],
                    mc_chain_checkpoint=mc_chain_checkpoint_current)
    

    
  


