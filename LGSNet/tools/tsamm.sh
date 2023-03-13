CUDA_VISIBLE_DEVICES=2 python /home/yww/LGSNET/LGSNet/tools/main.py --cfg '/home/yww/LGSNET/LGSNet/experiments/samm.yaml' \
                                                 --dataset 'samm'\
                                                 --subject 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 24 25 26 28 30 32 33 34 35 36 37 99
python /home/yww/1_spot/MSA-Net/tools/F1_score_last.py --path '/home/yww/LGSNET/LGSNet/output/samm' \
                                                                --most_pos_num 14 --start_threshold 100 \
                                                                --ann '/home/yww/LGSNET/LGSNet/samm_annotation_sample.csv'\
                                                                --dataset 'samm'\
                                                                --label_frequency 6.67 --base_duration 15
                                                                