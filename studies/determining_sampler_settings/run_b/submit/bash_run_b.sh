jid0=$(sbatch run_b/submit/analysis_run_b_0.sh)
squeue -u $USER -o "%u %.10j %.8A %.4C %.40E %R"
