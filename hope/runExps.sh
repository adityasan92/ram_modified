#!/bin/bash
for ((i=0; i<=2; i++)); do
   
   ###################################################################################################
   ### To reproduce, change the script parameters accordingly and uncomment the corresponding line ###
   ###################################################################################################
 
   #nohup python ram_vanilla_new.py vanilla_new_untrans_$i >> vn_u_$i.log &   
   #nohup python ram_srt_new.py dropout_0d25_reward_new_untrans_$i >> d0d25_r_u_$i.log &   
   nohup python ram_srt_new.py dropout_0d25_noReward_new_untrans_$i >> d0d25_nr_u_$i.log &   
   
   
   #nohup python ram_vanilla_new.py vanilla_new_$i >> vn_$i.log &   
   #nohup python ram_srt_new.py dropout_0d75_noReward_new_$i >> d0d75_nr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d25_noReward_new_$i >> d0d25_nr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d25_reward_new_$i >> d0d25_pr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d75_reward_new_$i >> d0d75_pr_$i.log &   
   #nohup python ram_srt_new.py mg_reward_new_$i >> mg_pr_$i.log &   
   #nohup python ram_srt_new.py mg_noReward_new_$i >> mg_nr_$i.log &   
   #nohup python ram_concrete_dropout_new.py concrete_reward_new_$i >> conc_r_$i.log &
   #nohup python ram_concrete_dropout_new.py concrete_noReward_new_$i >> conc_nr_$i.log &

   #####

done


