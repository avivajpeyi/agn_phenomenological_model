import lal
import lalsimulation as lalsim
import matplotlib.pyplot as plt
import numpy as np
plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")



def get_fseries_and_tseries(fmin = 20, fmax = 2048, duration = 8.):
    deltaF = 1./duration
    fseries = np.linspace(fmin, fmax, num=int((fmax-fmin)/deltaF)+1)
    tseries = np.linspace(0, duration, num=int((fmax-fmin)/deltaF)*2)
    return fseries, tseries


def get_strain(fseries, fref=100):
    # injection param
    m1, m2 = 38.90726199927476, 4.099826620277696
    m1*=lal.MSUN_SI
    m2*=lal.MSUN_SI
    s1x, s1y, s1z = -0.5292121532005147, 0.0815506948762848, 0.6489430710417405
    s2x, s2y, s2z = 0.32082521678503834, -0.7843006704918378, 0.02983346070373225
    iota = 2.489741666120003
    phase = 2.3487991630017353
    dist= 100


    # convert fseries to lal vector 
    F = fseries
    F = lal.CreateREAL8Vector(len(F))
    F.data[:] =  fseries

    # compute strain 
    WFdict = lal.CreateDict()
    hplus, hcross = lalsim.SimInspiralChooseFDWaveformSequence(
        phase, m1, m2,
        s1x, s1y, s1z,
        s2x, s2y, s2z,
        fref,
        dist * 1e6 * lal.PC_SI, iota, WFdict,
        lalsim.IMRPhenomXPHM, F
    )

    return dict(
        asd = np.abs(hplus.data.data),
        phase = np.unwrap(np.angle(hplus.data.data)),
        time = np.fft.irfft(hplus.data.data)
    )


def plot_strain_at_different_fref(frefs=[]):
    fseries, tseries = get_fseries_and_tseries(duration=8)
    fig, ax = plt.subplots(2,1, figsize=(6,7))

    for fref in frefs:
        strain = get_strain(fseries, fref)
        label = r"$f_{\rm  ref} = " + str(fref) + "$ Hz"
        ax[0].loglog(fseries, strain['asd'], label=label, alpha=0.75)
        ax[1].plot(tseries, strain['time'], label=label, alpha=0.75)

    ax[0].set_xlabel('Freq [Hz]')
    ax[1].set_xlabel('Time [s]')
    ax[0].set_ylabel(r'ASD [Hz$^{-1/2}$]')
    ax[1].set_ylabel('Strain')
    ax[0].legend()
    plt.tight_layout()
    plt.savefig('comparing_singal_at_different_fref.png')


if __name__ == '__main__':
    plot_strain_at_different_fref([0.001])
    