draw_population_samples --outdir pop_files --prior-file ../data/bbh.prior
pbilby_pe_jobgen --pop-files pop_files/pop_a_highsnr.dat pop_files/pop_b_highsnr.dat --prior-file ../data/bbh.prior --psd-file ../data/aLIGO_late_psd.txt

cdf_check --pop-a-regex "/fred/oz980/avajpeyi/projects/agn_phenomenological_model/studies/high_snr_multiple_population/outdir_pop_a_highsnr/out_pop_a_highsnr_*/result/pop*highsnr_*.json"