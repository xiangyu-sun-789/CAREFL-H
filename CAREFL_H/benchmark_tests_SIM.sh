sem=None
noise=None
n=-1


normalize_datasets=1
number_of_datasets=10
fix_AffineCL_forward=1
batch_size=-1

declare -a dataset_types=("SIM_benchmark_SIM" "SIM_benchmark_SIM-c" "SIM_benchmark_SIM-G" "SIM_benchmark_SIM-ln")

declare -a splits=(0.8 1.0)

# alternative hyperparameter values, without default values
declare -a epochs=(500 1000 2000)
declare -a nhs=(2 10 20)
declare -a nls=(1 7 10)
declare -a priors=("gaussian")
declare -a weight_decays=(0.0001 0.001 0.1)


# hyperparameters with default values
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
  echo $sem $noise $n $dataset_type $split
  python main_carefl_comparison_multiple_times.py --result_folder_suffix default --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done


# epochs
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
for epoch in "${epochs[@]}"; do
  echo $sem $noise $n $dataset_type $split $epoch
  python main_carefl_comparison_multiple_times.py --result_folder_suffix epochs --carefl_epochs $epoch --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done
done


# nhs
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
for nh in "${nhs[@]}"; do
  echo $sem $noise $n $dataset_type $split $nh
  python main_carefl_comparison_multiple_times.py --result_folder_suffix nhs --carefl_nh $nh --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done
done


# nls
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
for nl in "${nls[@]}"; do
  echo $sem $noise $n $dataset_type $split $nl
  python main_carefl_comparison_multiple_times.py --result_folder_suffix nls --carefl_nl $nl --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done
done


# priors
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
for prior in "${priors[@]}"; do
  echo $sem $noise $n $dataset_type $split $prior
  python main_carefl_comparison_multiple_times.py --result_folder_suffix priors --carefl_prior_dist $prior --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done
done


# weight_decays
for dataset_type in "${dataset_types[@]}"; do
for split in "${splits[@]}"; do
for weight_decay in "${weight_decays[@]}"; do
  echo $sem $noise $n $dataset_type $split $weight_decay
  python main_carefl_comparison_multiple_times.py --result_folder_suffix weight_decays --carefl_weight_decay $weight_decay --split $split --SEM $sem --noise $noise --n $n --result_folder temp_results/results_one_hyperparameter/SIM/$dataset_type/ --dataset_type $dataset_type --normalize_datasets $normalize_datasets --number_of_datasets $number_of_datasets --fix_AffineCL_forward $fix_AffineCL_forward --batch_size $batch_size
done
done
done
