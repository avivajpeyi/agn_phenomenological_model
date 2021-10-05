from agn_utils.bh_evolver.backwards import PrecessionBackwardEvolver

from bilby.gw.prior import  BBHPriorDict
import pandas as pd

def  test_back_evolver():
    bbh = BBHPriorDict().sample(1)
    bbh = {k:float(v) for k, v in bbh.items()}
    print(f"Conveting BBH")
    evol = PrecessionBackwardEvolver(bbh, fref=20)
    converted = evol.run_evolver()
    print("Finished conversion!")
    print(pd.DataFrame([bbh, converted]).T)


if __name__ == '__main__':
    test_back_evolver()