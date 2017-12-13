#!/bin/bash
for ((i=0; i<=9; i++)); do
   ### To reproduce, change the script parameters accordingly and uncomment the corresponding line ###
   #nohup python ram_srt.py dropout_0d25_plusReward_eta4_$i >> d0d25pre4_$i.log &   
   #nohup python ram_srt.py dropout_0d25_noReward_eta4_$i >> d0d25pne4_$i.log &   
   #nohup python ram_srt.py dropout_0d5_plusReward_eta4_$i >> d0d5pre4_$i.log &   
   #nohup python ram_srt.py dropout_0d5_noReward_eta4_$i >> d0d5pne4_$i.log &   
   nohup python ram_srt.py dropout_0d75_noReward_eta4_$i >> d0d75pne4_$i.log &   
   #nohup python ram_srt.py dropout_0d75_plusReward_eta4_$i >> d0d75pre4_$i.log &   
   #nohup python ram_srt.py multGauss_sigma1_plusReward_eta4_$i >> mg1spre4_$i.log &   
   #nohup python ram_srt.py multGauss_sigma1_noReward_eta4_$i >> mg1snre4_$i.log &   
   #nohup python ram_concrete_dropout.py concrete_dropout_plusReward_eta4_$i >> cre4_$i.log &   
   #nohup python ram_concrete_dropout.py concrete_dropout_noReward_eta4_$i >> cne4_$i.log &   
done


