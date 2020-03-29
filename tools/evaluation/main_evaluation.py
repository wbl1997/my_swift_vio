```
Test the algorithms with many missions of data.

input: euroc dataset dir, uzh-fpv dataset dir
data sesssions for evaluation
dir to rpg_evaluation_tool
dir to msckf config file
regression parameter to tune

output: msckf results organized as required by rpg evaluation tool
Then call rpg evaluation tool to draw plots and get metrics
Grab the results from rpg evaluation results to get a brief report on performance.

Important parameters
python2 scripts/analyze_trajectory_single.py results/euroc_mono_stereo/laptop/vio_mono/laptop_vio_mono_MH_05
--recalculate_errors True 
--mul_trials
--est_type traj_est


```
