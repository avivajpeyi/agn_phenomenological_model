draw_population_samples --outdir pop_files --prior-file comoving_bbh.prior

pbilby_pe_jobgen --pop-files pop_files/pop_a_highsnr.dat --prior-file comoving_bbh.prior --psd-file ../data/aLIGO_late_psd.txt

cdf_check --pop-a-regex "outdir/pop_a/*.json" --pop-b-regex "outdir/pop_b/*.json" 


for i in range(0,40):
    print(f"parallel_bilby_generation pop_b_highsnr_{i:02}.ini")

