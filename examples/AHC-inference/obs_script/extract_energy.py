import numpy as np

def get_energy(path_,Ne, data_num = None, nstart=100):
    print("Getting energy from path:", path_)
    overall_factor = np.loadtxt(path_ + '/overall_factor.txt')
    data = np.loadtxt(path_ + '/data_inference.txt'.format(data_num), skiprows=1)
    etot = data[:,7]*2*overall_factor/Ne
    ebp = 2.0/Ne*((data[:,1]+data[:,3]+data[:,4]+data[:,5]+data[:,6])*overall_factor - data[:,4])
    evar = data[:,10]*2*overall_factor/Ne
    eint= 2.0/Ne*overall_factor*(data[:,2])
    
    etot = etot[nstart:]
    evar = evar[nstart:]
    ebp = ebp[nstart:]
    eint = eint[nstart:]

    etot_avg = np.mean(etot)
    d2 = abs(etot - etot_avg)**2  # abs is for complex `a`
    etot_std = np.sqrt( d2.sum() / ((len(etot)-1)*len(etot)) )

    evar_avg = np.mean(evar)
    d2var = abs(evar - evar_avg)**2  # abs is for complex `a`
    evar_std = np.sqrt( d2var.sum() / ((len(evar)-1)*len(evar)) )


    ebp_avg = np.mean(ebp)
    d2bp = abs(ebp - ebp_avg)**2  # abs is for complex `a`
    ebp_std = np.sqrt( d2bp.sum() / ((len(ebp)-1)*len(ebp)) )

    eint_avg = np.mean(eint)
    d2int = abs(eint - eint_avg)**2  # abs is for complex `a`
    eint_std = np.sqrt( d2int.sum() / ((len(eint)-1)*len(eint)) )
    
    
    return etot_avg, etot_std, ebp_avg, ebp_std, evar_avg, evar_std, eint_avg, eint_std #2.0/Ne*overall_factor*np.mean(data[:,2][nstart:])
