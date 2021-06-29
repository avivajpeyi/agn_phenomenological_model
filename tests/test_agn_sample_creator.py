import os
import shutil
import unittest
import warnings

warnings.filterwarnings("ignore")

from agn_utils.create_agn_samples import (
    load_training_data,
    save_agn_samples_for_many_populations,
)
from agn_utils.plotting.overlaid_corner_plotter import overlaid_corner

class AGNsampleCreatorTest(unittest.TestCase):
    def setUp(self):
        self.outdir = "testoutdir"
        os.makedirs(self.outdir, exist_ok=True)

    # def tearDown(self):
    #     if os.path.exists(self.outdir):
    #         shutil.rmtree(self.outdir)

    def test_something(self):
        fname = os.path.join(self.outdir, "posteriors_list.h5")
        save_agn_samples_for_many_populations(num=1, fname=fname)
        df = load_training_data(fname)
        print(df.describe())
        params = ['chi_p', 'chi_eff', 'cos_tilt_1', 'cos_theta_12', 'mass_ratio']
        overlaid_corner(samples_list=[df], sample_labels=['test samples'], params=params, samples_colors=['red'], fname=fname.replace('.h5','.png'))
        self.assertTrue(os.path.isfile(fname))


if __name__ == "__main__":
    unittest.main()
