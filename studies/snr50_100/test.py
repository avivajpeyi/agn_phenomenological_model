import bilby
from pprint import pprint
from agn_utils.plot_corners import result_post_processing, generate_corner
from agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check import simple_violin_plotter, get_bilby_results, pe_cdf

TruePopA = "cornflowerblue"
TruePopB = "bisque"
DataA = "midnightblue"
DataB = "darkorange"

# f = 'outdir/pop_a/pop_a_highsnr_00_0_result.json'
# r = bilby.gw.result.CBCResult.from_json(filename=f)
# pprint(r.injection_parameters)
#
#
# r2 = result_post_processing(r)
# pprint(r2.injection_parameters)

#%%

pop_a_regex = 'outdir/pop_a/*.json'
pop_b_regex = 'outdir/pop_b/*.json'
# pop_a_pkl = 'pop_a.pkl'
# dat_a = get_bilby_results(pop_a_regex, pop_a_pkl, ["cos_tilt_1", "cos_theta_12", "chirp_mass"])
# simple_violin_plotter(dat_a, "pop_a_pe.png")

# rweight samples


pe_cdf(
    pops_regexs=dict(
        pop_a=pop_a_regex,
        pop_b=pop_b_regex
    ),
    true_pop_params=dict(
        popA=[0.5, 3],
        popB=[1, 0.25]),
    title="Population Comparision", fname="pop_compare_cdf.png",
    colors1=[TruePopA, TruePopB, DataA,  DataB],
    colors2=[TruePopA, TruePopB, DataA, DataB]
)



