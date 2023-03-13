CUDA_VISIBLE_DEVICES=1 python /home/yww/LGSNET/LGSNet/tools/main.py --cfg '/home/yww/1_spot/LGSNet/experiments/cas.yaml'\
     --dataset 'cas(me)^2' \
     --subject 16 15 19 20 21 22 23 24 25 26 27 29 30 31 32 33 34 35 36 37 38 40

python /home/yww/1_spot/MSA-Net/tools/F1_score_last.py --path '/home/yww/LGSNET/LGSNet/output/cas(me)^2_new' \
    --most_pos_num 14 --start_threshold 300 \
    --ann '/home/yww/LGSNET/LGSNet/casme2_annotation.csv' \
    --dataset 'cas(me)^2' \
    --label_frequency 1.0 \
    --base_duration 15