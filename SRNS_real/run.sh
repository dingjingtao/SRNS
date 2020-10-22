#!/bin/bash
F=32
for alpha in 5.0
do
    for warmup in 50
    do
        for tau in 10.0
        do
            for S1 in 8
            do
                for S2_div_S1 in 8
                do
                    for trial_id in 0
                    do
                        python SRNS_main.py --process SRNS-Final \
                        --S1 $S1 \
                        --lr 0.001 \
                        --regs 0.01 \
                        --alpha $alpha \
                        --warmup $warmup \
                        --embedding_size $F \
                        --S2_div_S1 $S2_div_S1 \
                        --temperature $tau \
                        --gpu 1 \
						--save_model \
                        --trial_id $trial_id \
                        --early_stop 100 \
                        --epoch 400
                    done
                done
            done
        done
    done
done
