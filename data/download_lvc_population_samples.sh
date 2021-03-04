wget https://dcc.ligo.org/public/0171/P2000434/001/Population_Samples.tar.gz -q
tar -xzf Population_Samples.tar.gz
mkdir lvc_popinf
mv Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json lvc_popinf/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json
rm -rf Population_Samples/
rm Population_Samples.tar.gz
