#!/bin/bash
for ((i=0; i<=2; i++)); do
   
   ###################################################################################################
   ### To reproduce, change the script parameters accordingly and uncomment the corresponding line ###
   ###################################################################################################
   
   #nohup python ram_vanilla_new.py vanilla_new_$i >> vn_$i.log &   
   #nohup python ram_srt_new.py dropout_0d75_noReward_new_$i >> d0d75_nr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d25_noReward_new_$i >> d0d25_nr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d25_reward_new_$i >> d0d25_pr_$i.log &   
   #nohup python ram_srt_new.py dropout_0d75_reward_new_$i >> d0d75_pr_$i.log &   
   nohup python ram_concrete_dropout_new.py concrete_reward_new_$i >> conc_r_$i.log &


   #nohup python ram_srt.py dropout_0d25_plusReward_eta4_$i >> d0d25pre4_$i.log &   
   #nohup python ram_srt.py dropout_0d25_noReward_eta4_$i >> d0d25pne4_$i.log &   
   #nohup python ram_srt.py dropout_0d5_plusReward_eta4_$i >> d0d5pre4_$i.log &   
   #nohup python ram_srt.py dropout_0d5_noReward_eta4_$i >> d0d5pne4_$i.log &   
   #nohup python ram_srt.py dropout_0d75_noReward_eta4_$i >> d0d75pne4_$i.log &   
   #nohup python ram_srt.py dropout_0d75_plusReward_eta4_$i >> d0d75pre4_$i.log &   
   #nohup python ram_srt.py multGauss_sigma1_plusReward_eta4_$i >> mg1spre4_$i.log &   
   #nohup python ram_srt.py multGauss_sigma1_noReward_eta4_$i >> mg1snre4_$i.log &   
   #nohup python ram_concrete_dropout.py concrete_dropout_plusReward_eta4_$i >> cre4_$i.log &   
   #nohup python ram_concrete_dropout.py concrete_dropout_noReward_eta4_$i >> cne4_$i.log &   
   #nohup python ram_srt.py vanilla_translated_$i >> vanilla_$i.log &   
   #nohup python ram_srt.py vanilla_untranslated_$i >> vanilla_un_$i.log &   

done


