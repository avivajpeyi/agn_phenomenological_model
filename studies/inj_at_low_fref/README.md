
pbilby_pe_jobgen --pop-files pop_files/pop_a_validsnr.dat --prior-file comoving_bbh.prior --psd-file ../data/aLIGO_late_psd.txt --cluster sstar --fref 0.001
pbilby_pe_jobgen --pop-files pop_files/pop_b_validsnr.dat --prior-file comoving_bbh.prior --psd-file ../data/aLIGO_late_psd.txt --cluster gstar --fref 0.001

cdf_check --pop-a-regex "outdir/pop_a/*.json" --pop-b-regex "outdir/pop_b/*.json" 

