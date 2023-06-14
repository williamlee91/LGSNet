# python /home/yww/1_spot/samm_sampler.py --new_img_path '/home/yww/1_spot/samm_merge_part_2000'\
#                                     --flow_path  '/home/yww/1_spot/samm_merge_part_flow_2000'\
#                                     --save_path_img '/home/yww/1_spot/samm_img_merge_part_sample_2000'\
#                                     --save_path_flow '/home/yww/1_spot/samm_flow_merge_part_sample_2000'\
#                                     --frequency 6.667 \
#                                     --ann_ori '/home/yww/1_spot/samm_annotation_L800.csv'\
#                                     --new_ann_path '/home/yww/1_spot/samm_annotation_merge_part_2000_L800_new2.csv'\
#                                     --is_sample True\
#                                     --sliding_window 2000

CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'rgb'\
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/rgb_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped'\
                            --output_dir '/home/yww/WTAL/CAS_feature_8/rgb'\
                            --batch_size 64 \
                            --frequency 8\
                            --chunk_size 8\
                            --dataset 'cas'

CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'flow' \
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/flow_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/cas(me)2_flow'\
                            --output_dir '/home/yww/WTAL/CAS_feature_8/flow'\
                            --batch_size 64 \
                            --frequency 8\
                            --chunk_size 8\
                            --dataset 'cas'


CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'rgb'\
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/rgb_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped'\
                            --output_dir '/home/yww/WTAL/CAS_feature_4/rgb'\
                            --batch_size 64 \
                            --frequency 4\
                            --chunk_size 4\
                            --dataset 'cas'

CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'flow' \
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/flow_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/cas(me)2_flow'\
                            --output_dir '/home/yww/WTAL/CAS_feature_4/flow'\
                            --batch_size 64 \
                            --frequency 4\
                            --chunk_size 4\
                            --dataset 'cas'

# python /home/yww/1_spot/sam_split.py 

CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'rgb'\
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/rgb_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/CAS(ME)2_longVideoFaceCropped/longVideoFaceCropped'\
                            --output_dir '/home/yww/WTAL/CAS_feature_16/rgb'\
                            --batch_size 16 \
                            --frequency 16\
                            --chunk_size 16\
                            --dataset 'cas'

CUDA_VISIBLE_DEVICES=3 python /home/yww/1_spot/pytorch-i3d-feature-extraction/extract_features.py --mode 'flow' \
                            --load_model '/home/yww/1_spot/pytorch-i3d-feature-extraction/models/flow_imagenet.pt'\
                            --input_dir '/home/yww/1_spot/cas(me)2_flow'\
                            --output_dir '/home/yww/WTAL/CAS_feature_16/flow'\
                            --batch_size 16 \
                            --frequency 16\
                            --chunk_size 16\
                            --dataset 'cas'
