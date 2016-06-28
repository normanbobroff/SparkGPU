#!/bin/bash

#   --master local[16] \
#  --total-executor-cores 100 \

./bin/spark-submit \
  --class "javaPCA" \
  --master spark://localhost \
  target/pca-0.0.1-SNAPSHOT.jar
