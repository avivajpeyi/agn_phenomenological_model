import glob

sub_file_regex = "out*/sub*/anal*.sh"

sub_files = glob.glob(sub_file_regex)

for f in sub_files:
    content = open(f, 'r').read()
    content = content.replace("mem-per-cpu=2000", "mem-per-cpu=4000")
    open(f, 'w').write(content)
