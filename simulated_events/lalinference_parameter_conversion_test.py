from pprint import pprint

import lalsimulation
import pandas as pd

MSUN = 1.9884099021470415e+30  # Kg
REFERENCE_FREQ = 20.0


def transform_precessing_spins(theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1,
                               a_2, mass_1, mass_2, reference_frequency, phase):
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = (
        lalsimulation.SimInspiralTransformPrecessingNewInitialConditions(
            theta_jn, phi_jl, tilt_1, tilt_2,
            phi_12, a_1, a_2,
            mass_1 * MSUN, mass_2 * MSUN,
            reference_frequency, phase))
    return iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z


def transform_component_spins(iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y,
                              spin_2z, mass_1, mass_2, phase, reference_frequency):
    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = (
        lalsimulation.SimInspiralTransformPrecessingWvf2PE(
            incl=iota,
            S1x=spin_1x, S1y=spin_1y, S1z=spin_1z,
            S2x=spin_2x, S2y=spin_2y, S2z=spin_2z,
            m1=mass_1, m2=mass_2,
            fRef=reference_frequency, phiRef=phase
        ))
    return theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2


def main():
    init_params = dict(
        mass_1=60.0, mass_2=50.0,
        spin_1x=0.32,
        spin_1y=-0.32,
        spin_1z=0.32,
        spin_2x=0.00,
        spin_2y=0.0,
        spin_2z=1.00,
        phase=0.0, iota=0.0,
        reference_frequency=REFERENCE_FREQ
    )

    theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2 = \
        transform_component_spins(**init_params)

    precessing_spin_params = dict(
        theta_jn=theta_jn, phi_jl=phi_jl,
        tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        a_1=a_1, a_2=a_2,
        mass_1=init_params['mass_1'], mass_2=init_params['mass_2'],
        reference_frequency=init_params["reference_frequency"],
        phase=init_params["phase"]
    )

    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = \
        transform_precessing_spins(**precessing_spin_params)

    final_params = dict(
        iota=iota, spin_1x=spin_1x, spin_1y=spin_1y, spin_1z=spin_1z, spin_2x=spin_2x,
        spin_2y=spin_2y, spin_2z=spin_2z
    )

    print("Start:")
    pprint(init_params)
    print("Converted to Spherical-spins:")
    pprint(precessing_spin_params)
    print("Converted to Cartesian-spins:")
    pprint(final_params)

    print("\nCompare init with final:")
    for i in ['mass_1', 'mass_2', 'reference_frequency', 'phase']:
        init_params.pop(i)
    init_params['label'] = "INIT"
    final_params['label'] = "FINAL"
    df = pd.DataFrame([init_params, final_params])
    df = df.set_index('label', drop=True)
    print(df.transpose())


if __name__ == "__main__":
    main()
