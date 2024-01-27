# keepflow
This repository aims to provide the code base for temporal prediction tasks, such as trajectory, motion, and video predictions.

## News
- Jan 2024: repository public on GitHub :confetti_ball::confetti_ball::confetti_ball:

## Dependencies
```
pip install -r requirements.txt
git submodule init
git submodule update
```

## Data preprocessing
### Trajectory prediction
#### datasets supported by [trajdata](https://github.com/NVlabs/trajdata) such as ETH/UCY and Stanford Drone Datasets.
  
Please follow the [trajdata instruction](https://github.com/NVlabs/trajdata/blob/main/DATASETS.md) and specify the dataset path by cfg.DATA.PATH in keepflow/utils/default_params.py.

- ETH/UCY
```
cfg.DATA.PATH/traj/raw/all_data
            ├── biwi_eth.txt
            ├── biwi_hotel.txt
            ├── crowds_zara01.txt
            ├── crowds_zara02.txt
            ├── crowds_zara03.txt
            ├── students001.txt
            ├── students003.txt
            └── uni_examples.txt
```

#### JackRabbot Dataset and Benchmark Trajectory Forecasting

Please follow the [jrdb-traj instruction](https://github.com/vita-epfl/JRDB-Traj) and execute `dataload.sh` and `preprocess.sh` in extern/traj/jrdb-traj/.


### Video prediction

TBA

### Motion prediction

TBA

## Supported Models
### Trajectory prediction
- SocialLSTM based on [socialGAN GitHub Code](https://github.com/agrimgupta92/sgan)
- Trajectron++ based on [Trajectron++ GitHub Code](https://github.com/StanfordASL/Trajectron-plus-plus)
- Motion Indeterminacy Diffusion based on [MID GitHub Code](https://github.com/Gutianpei/MID)
- FlowChain based on [FlowChain GitHub Code](https://github.com/meaten/FlowChain-ICCV2023)

## Training
```
python train.py --config_file CONFIG_FILE_PATH --device DEVICE
```
Example
```
python train.py --config_file configs/traj/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --device cuda:0
```
You can execute training consecutively by script
```
python scripts/run_configs.py --config_dir configs/traj/FlowChain/CIF_separate_cond_v_trajectron/ --device cuda:0 (--test_only)
```

## Evaluation
```
python test.py --config_file CONFIG_FILE_PATH --device DEVICE (--visualize)
```
Example
```
python test.py --config_file configs/traj/FlowChain/CIF_separate_cond_v_trajectron/eth.yml --device cuda:0
```



