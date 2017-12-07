#!/bin/bash
for i in $(seq $1 1 $2); do
   echo "Killing $i"
   kill $i
done
