

for regs in 0.0
do
    for F in 32
    do
        for trial_id in '0' 
        do
            for lr in 0.001
            do
                python main.py --model 'uniform'\
				               --embedding_size $F \
							   --dataset 1\
                                  --epochs 400 \
                                  --early_stop 400 \
                                  --lr $lr \
                                  --regs $regs \
                                  --trial_id $trial_id\
                                  --gpu '2'
            done
        done
    done
done
