import pesummary
from pesummary.io import read

data = read("/home/zoheyr.doctor/public_html/O3/O3aCatalog/data_release/all_posterior_samples/GW190517_055101.h5")
print(data.summary)
