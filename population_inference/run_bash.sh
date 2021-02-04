#gwpopulation_pipe agn_gwpop_inference.ini
#gwpopulation_pipe ind_gwpop_inference.ini



#gwpopulation_pipe_collection /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/ind_outdir/ind_config_complete.ini &\
#gwpopulation_pipe_collection /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_outdir/agn_config_complete.ini

gwpopulation_pipe_analysis /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_outdir/agn_config_complete.ini --prior /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_agn_tilt_powerlaw_redshift.prior --label agn_mass_c_iid_mag_agn_tilt_powerlaw_redshift --models SmoothedMassDistribution --models iid_spin_magnitude --models agn_spin_orientation --models gwpopulation.models.redshift.PowerLawRedshift --vt-models SmoothedMassDistribution --vt-models gwpopulation.models.redshift.PowerLawRedshift &\
gwpopulation_pipe_analysis /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/ind_outdir/ind_config_complete.ini --prior /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/priors/mass_c_iid_mag_ind_tilt_powerlaw_redshift.prior --label ind_mass_c_iid_mag_ind_tilt_powerlaw_redshift --models SmoothedMassDistribution --models iid_spin_magnitude --models ind_spin_orientation --models gwpopulation.models.redshift.PowerLawRedshift --vt-models SmoothedMassDistribution --vt-models gwpopulation.models.redshift.PowerLawRedshift
