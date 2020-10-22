for alpha in 10.0
do
    for S in 20
    do
        for T0 in 50
        do
            for sigma in 2
            do
                for trial_id in '0'
                do
                    python main.py --model 'SRNS'\
                                  --T0 $T0 \
								  --dataset 2\
                                  --dynamic_alpha 'increase' \
                                  --S1 $S\
                                  --S2 $S\
                                  --lr 0.001\
                                  --regs 0.0\
                                  --epochs 400 \
                                  --early_stop 400 \
                                  --alpha $alpha\
                                  --fn_num 1 \
                                  --sigma $sigma \
                                  --gpu '3'\
                                  --trial_id $trial_id
                done
            done
        done
    done
done






