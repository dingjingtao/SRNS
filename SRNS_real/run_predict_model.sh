#!/bin/bash


for model_file in 'uniform' 'enmf' 'nncf' 'aobpr' 'irgan' 'rns' 'advir' 'srns'
do
    python model_predict.py --process 'model_predict' \
                    --use_pretrain \
					--model_file $model_file
done

