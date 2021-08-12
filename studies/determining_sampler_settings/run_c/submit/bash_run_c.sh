jid0=$(sbatch run_c/submit/analysis_run_c_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
