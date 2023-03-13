CUDA_VISIBLE_DEVICES=2 python /home/yww/LGSNET/LGSNet/tools/main.py --cfg '/home/yww/1_spot/LGSNet/experiments/cas3.yaml'\
      --dataset 'cas(me)^3' 

python /home/yww/1_spot/MSA-Net/tools/F1_score_last.py --path '/home/yww/LGSNET/LGSNet/output/cas(me)^3' \
    --most_pos_num 14 --start_threshold 300 \
    --ann '/home/yww/LGSNET/LGSNet/cas3_annotation_full_me_reduce.csv' \
    --dataset 'cas(me)^3' \
    --label_frequency 1.0 \
    --base_duration 15