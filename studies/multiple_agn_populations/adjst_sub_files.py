import glob
from tqdm.auto import tqdm

sub_file_regex = "out*/out*/sub*/anal*.sh"

sub_files = glob.glob(sub_file_regex)

for f in tqdm(sub_files):
    content = open(f, 'r').read()
    # content = content.replace("ntasks-per-node=16", "ntasks-per-node=14")
    # content = content.replace("MPI_PER_NODE=16", "MPI_PER_NODE=14")
    content = content.replace("mem-per-cpu=4000", "mem-per-cpu=2000")
    content = content.replace("time=24:00:00", "time=160:00:00")
    open(f, 'w').write(content)
