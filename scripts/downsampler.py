import glob, tqdm, time
import pandas as pd
from pprint import pprint

MAX_SAMPLES = 5000

def main():

    fnames = []
    lens = []
    files = glob.glob("../data/gwtc_samples/*.csv")
    for f in tqdm.tqdm(files):
        try:
            lens.append(len(pd.read_csv(f, sep=" ")))
            fnames.append(f)
        except Exception as e:
            print(f"ERROR {f}: {e}")

    samples_count = pd.DataFrame(dict(fnames=fnames,lens=lens))
    samples_count = samples_count[samples_count['lens']>MAX_SAMPLES]
    samples_count = samples_count.reset_index(drop=True)
    print(samples_count)


    for i in range(len(samples_count)):
        fname = samples_count.iloc[i]['fnames']
        df = pd.read_csv(fname, sep=" ").sample(MAX_SAMPLES)
        df.to_csv(fname, sep=' ', index=False)
        print(f"{fname} len {samples_count.iloc[i]['lens']}-->{MAX_SAMPLES}" )
        
if __name__=="__main__":
    main()