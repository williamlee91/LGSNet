
# LGSNet: A Two-stream Network for Micro- and Macro-expression Spotting with Background Modeling 

Micro- and macro-expression spotting in an untrimmed video is a challenging task, due to the mass generation of false positive samples. 
Most existing methods localize higher response areas by extracting hand-crafted features or cropping specific regions from all or some key raw images.
However, these methods either neglect the continuous temporal information or model the inherent human motion paradigms (background) as foreground. 
% In this work, we argue that there are some important issues to be addressed, including simplification of the encoding process, 
% efficient exploitation of continuous temporal information, reduction of interference from background intervals, 
% refinement of short interval localization, and removal of unrealistic noisy data.
Consequently, we propose a novel two-stream network, named Local suppression and Global enhancement Spotting Network (LGSNet), which takes segment-level features 
from optical flow and videos as input.
LGSNet adopts anchors to encode expression intervals and selects the encoded deviations as the object of optimization.
Furthermore, we introduce a Temporal Multi-Receptive Field Feature Fusion Module (TMRF$^3$M) and a Local Suppression and Global Enhancement Module (LSGEM), 
which help spot short intervals more precisely and suppress background information.
To further highlight the differences between positive and negative samples, we set up a large number of random pseudo ground truth intervals (background clips) 
on some discarded sliding windows to accomplish background clips modeling to counteract the effect of non-expressive face and head movements.
Experimental results show that our proposed network achieves state-of-the-art performance on the CAS(ME)$^2$, CAS(ME)$^3$ and SAMM-LV datasets. 

## Requirements
* Create the anaconda environment as what we used.

```
conda env create -f environment.yaml
```

> 2022.12.21 

The first version!


> 2023.02.21 

The newest version! We have refined some parameters.

CAS(ME)^2: recall: 0.366947, precision: 0.629808, f1_score: 0.463717 

SAMM-LV: recall: 0.355 precision:0.429 f1_score: 0.388

> 2023.03.10

Codes will continue to be updated!

## Prepare Datasets
* Download the CAS(ME)^2, CAS(ME)^3 and SAMM-LV datasets.

* Generate annatations for all datasets.
```
python ./LGSNET/get_anatation_casme2.py
```
```
python ./LGSNET/get_anatation_casme3_reduce_more.py
```
```
python ./LGSNET/get_anatation_samm.py
```

* Reconstructe SAMM-LV datasets and then extract optical flow.
```
python ./LGSNET/samm_sampler.py
```

* extract optical flow for CAS(ME)^2 and CAS(ME)^3 datasets.
```
python ./LGSNET/extract_optical_flow.py
```

* Generate features for all datasets.

All paths inside needs to be changed.
```
sh ./LGSNET/zextract.sh
```

* Generate sliding windows for all datasets.
On the CAS(ME)^2 dataset.
```
python ./LGSNET/cas_split.py
```
On the CAS(ME)^3 dataset.
```
python ./LGSNET/cas3_split.py
```
On the SAMM-LV dataset.
```
python ./home/yww/LGSNET/sam_split.py
```

More details can be found on the following websites: 
>I3D Feature Extraction (https://github.com/Finspire13/pytorch-i3d-feature-extraction);

>MSA-Net (https://github.com/blowing-wind/MSA-Net);

>A2Net (https://github.com/VividLe/A2Net)

## Train models & Validate results
* Modify the parameters for the corresponding dataset in the experiments (./LGSNET/LGSNet/experiments)

* Run the train scripts:

On the CAS(ME)^2 dataset.
```
sh ./LGSNET/LGSNet/tools/tcas.sh
```
On the CAS(ME)^3 dataset.
```
sh ./LGSNET/LGSNet/tools/tcas3.sh
```
On the SAMM-LV dataset.
```
sh ./LGSNET/LGSNet/tools/tsamm.sh
```

## Validate available results
```

python /home/yww/1_spot/MSA-Net/tools/F1_score_last.py --path '/home/yww/LGSNET/LGSNet/output/cas(me)^2_new' \
    --most_pos_num 14 --start_threshold 300 \
    --ann '/home/yww/LGSNET/LGSNet/casme2_annotation.csv' \
    --dataset 'cas(me)^2' \
    --label_frequency 1.0 \
    --base_duration 15
```

## Citation

```
@article{yu2023lgsnet,
  title={LGSNet: A Two-Stream Network for Micro-and Macro-Expression Spotting With Background Modeling},
  author={Yu, Wang-Wang and Jiang, Jingwen and Yang, Kai-Fu and Yan, Hong-Mei and Li, Yong-Jie},
  journal={IEEE Transactions on Affective Computing},
  year={2023},
  publisher={IEEE}
}
```
## Contact
If you have any question or comment, please contact the first author of the paper - Wang-Wang Yu (yuwangwang91@163.com).
