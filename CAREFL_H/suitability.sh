sem="LSNM-tanh-exp-cosine"
noise="continuous-bernoulli"
python suitability.py --SEM $sem --noise $noise --result_folder temp_results/suitability/$sem/$noise/


sem="LSNM-sine-tanh"
noise="uniform"
python suitability.py --SEM $sem --noise $noise --result_folder temp_results/suitability/$sem/$noise/


sem="LSNM-sigmoid-sigmoid"
noise="exp"
python suitability.py --SEM $sem --noise $noise --result_folder temp_results/suitability/$sem/$noise/
