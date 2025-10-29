#!/bin/bash

SRC="/ds/images/MSCOCO2017/train2017"
DST="/netscratch/aslam/TCAV/PetImages/Concepts"
N=50

for SEED in {0..10}; do
    echo "Running with seed $SEED..."
    python generate-random-concept.py "$SRC" "$DST" "$N" "$SEED"
done
