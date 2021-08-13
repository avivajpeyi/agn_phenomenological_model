import bilby
from pprint import pprint
from agn_utils.plot_corners import result_post_processing, generate_corner
from agn_utils.plotting.cdf_plotter.sigma_cdf_difference_check import simple_violin_plotter, get_bilby_results, pe_cdf
from agn_utils.pe_postprocessing.posterior_reweighter import rejection_sample_population


TruePopA = "cornflowerblue"
TruePopB = "bisque"
DataA = "midnightblue"
DataB = "darkorange"

pop_a_regex = 'outdir/pop_a/*.json'
pop_b_regex = 'outdir/pop_b/*.json'
pop_a_pkl = 'pop_a.pkl'
pop_b_pkl = 'pop_a.pkl'
dat_a = get_bilby_results(pop_a_regex, pop_a_pkl, ["cos_tilt_1", "cos_theta_12", "chirp_mass"])
simple_violin_plotter(dat_a, "pop_a_pe.png")
pe_cdf(
    pops_dat_dicts=dict(
        pop_a=dat_a,
    ),
    true_pop_params=dict(
        popA=[0.5, 3]
    ),
    title="Population A", fname="pop_a_cdf.png",
    colors1=[TruePopA, DataA],
    colors2=[TruePopA, DataA]
)


dat_a["posteriors"] = rejection_sample_population(dat_a["posteriors"], true_population_param=dict(sigma_1=0.5, sigma_12=3))
simple_violin_plotter(dat_a, "pop_a_pe_reweighted.png")
pe_cdf(
    pops_dat_dicts=dict(
        pop_a=dat_a,
    ),
    true_pop_params=dict(
        popA=[0.5, 3]
    ),
    title="Population A Reweighted", fname="pop_a_cdf_reweighted.png",
    colors1=[TruePopA, DataA],
    colors2=[TruePopA, DataA]
)

dat_b = get_bilby_results(pop_b_regex, pop_b_pkl, ["cos_tilt_1", "cos_theta_12", "chirp_mass"])
pe_cdf(
    pops_dat_dicts=dict(
        pop_B=dat_b,
    ),
    true_pop_params=dict(
        popB=[1, 0.25]
    ),
    title="Population B", fname="pop_b_cdf.png",
    colors1=[TruePopB, DataB],
    colors2=[TruePopB, DataB]
)
dat_b["posteriors"] = rejection_sample_population(dat_b["posteriors"], true_population_param=dict(sigma_1=1, sigma_12=0.25))
pe_cdf(
    pops_dat_dicts=dict(
        pop_b=dat_b,
    ),
    true_pop_params=dict(
        popB=[1, 0.25]
    ),
    title="Population B", fname="pop_b_cdf_reweighted.png",
    colors1=[TruePopB, DataB],
    colors2=[TruePopB, DataB]
)
pe_cdf(
    pops_dat_dicts=dict(
        pop_a=dat_a,
        pop_b=dat_b
    ),
    true_pop_params=dict(),
    title="Population Comparision (reweighted)", fname="pop_compare_cdf_reweighted.png",
    colors1=[TruePopA, TruePopB, DataA,  DataB],
    colors2=[TruePopA, TruePopB, DataA, DataB]
)



