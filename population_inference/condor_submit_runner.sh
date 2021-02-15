#!/bin/bash
condor_submit_dag /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/agn_pop_outdir/submit/agn_pop.dag
condor_submit_dag /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/mixed_pop_outdir/submit/mixed_pop.dag
condor_submit_dag /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/lsc_outdir/submit/lsc.dag
condor_submit_dag /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/fixed_xi_0_pop_outdir/submit/fixed_xi_0_pop.dag
condor_submit_dag /home/avi.vajpeyi/projects/agn_phenomenological_model/population_inference/fixed_xi_1_pop_outdir/submit/fixed_xi_1_pop.dag