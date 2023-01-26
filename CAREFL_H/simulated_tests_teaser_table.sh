
normalize_datasets=1
number_of_datasets=10
fix_AffineCL_forward=1
batch_size=-1


#########################################################
# reproduce the results in the teaser table
#########################################################

n=10000
split=1.0
nl=4
nh=5

declare -a priors=("gaussian")
declare -a epochs=(750)
declare -a SEMs=("LSNM-sine-tanh")
declare -a noises=("uniform" "standard-gaussian")
declare -a g_magnitudes=(0.1 0.5 1 5 10)

for prior in "${priors[@]}"; do
for epoch in "${epochs[@]}"; do
for sem in "${SEMs[@]}"; do
for noise in "${noises[@]}"; do
for g_magnitude in "${g_magnitudes[@]}"; do
python main_carefl_comparison_multiple_times.py --method_to_test CAREFL --carefl_nl $nl --carefl_nh $nh --carefl_epochs $epoch --carefl_prior_dist $prior --g_magnitude $g_magnitude --split $split --SEM $sem --noise $noise --n $n --dataset_type simulated --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size --result_folder temp_results/simulated/vary_conditional_variance/CAREFL/split_$split/prior_$prior/$sem/$noise/epoch_$epoch/
# python main_carefl_comparison_multiple_times.py --method_to_test LOCI --carefl_nl -1 --carefl_nh -1 --carefl_epochs -1 --carefl_prior_dist None --g_magnitude $g_magnitude --split $split --SEM $sem --noise $noise --n $n --dataset_type simulated --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size --result_folder temp_results/simulated/vary_conditional_variance/LOCI/split_$split/$sem/$noise/
done
done
done
done
done
