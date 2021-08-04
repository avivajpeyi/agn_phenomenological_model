
draw_population_samples --outdir data --prior-file ../studies/high_snr_multiple_population/data/bbh.prior

pbilby_pe_jobgen --pop-files data/pop_a_highsnr.dat data/pop_b_highsnr.dat --prior-file ../studies/high_snr_multiple_population/data/bbh.prior --psd-file ../studies/high_snr_multiple_population/data/aLIGO_late_psd.txt


cdf_check --pop-a-regex "/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/high_snr_multiple_population/outdir_pop_a_highsnr/out_pop_a_highsnr_*/result/pop*highsnr_*.json"