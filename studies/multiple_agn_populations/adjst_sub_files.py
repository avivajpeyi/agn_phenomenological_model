import glob
from tqdm.auto import tqdm

sub_file_regex = "out*/out*/sub*/anal*.sh"

sub_files = glob.glob(sub_file_regex)

for f in tqdm(sub_files):
    content = open(f, 'r').read()
    # SBATCH --ntasks-per-node=16
    content = content.replace("ntasks-per-node=14", "ntasks=16")
    content = content.replace("MPI_PER_NODE=14", "MPI_PER_NODE=16")
    content = content.replace("mem-per-cpu=8000", "mem-per-cpu=2000")
    open(f, 'w').write(content)
