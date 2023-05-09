# ST$^2$: Spatial-Temporal State Transformer for Crowd-Aware Autonomous Navigation  

## Abstract
Empowering an intelligent agent with the ability of autonomous navigation in complex and dynamic environments is an important and active research topic in embodied artificial intelligence. In this letter, we address this challenging task from the view of exploiting both the spatial and temporal states of a mobile robot interacting with the crowded environment. Specifically, we propose a Spatial-Temporal State Transformer (ST$^2$) to encode the states while leveraging the deep reinforcement learning method to find the optimal navigation policy accordingly. Technically, the proposed ST$^2$ model consists of a global spatial state encoder and a temporal state encoder, which are built upon the Transformer structure. The spatial state encoder is devised to extract the global spatial features and capture the spatial interaction between pedestrians and the robot. The temporal state encoder is designed to model the temporal correlation among consecutive frames and infer the dynamic relationship of the spatial position transformation. Based on the comprehensive spatial-temporal state representation, the value-based reinforcement learning method is leveraged to obtain the optimal navigation policy. Extensive experiments demonstrate the superiority of the proposed ST$^2$ over representative state-of-the-art methods. The source code will be made publicly available.  

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository is organized in two parts: gym_crowd/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy ST2	
```
2. Test policies with 500 test cases.
```
python test.py --policy orca --phase test
python test.py --policy ST2--model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy orca --phase test --visualize --test_case 0
python test.py --policy ST2 --model_dir data/output --phase test --visualize --test_case 0
```
4. Visualize a test case.
```
python test.py --policy ST2 --model_dir data/output --phase test --visualize --test_case 0
```
## Citation
If you find the codes or paper useful for your research, please cite our paper:
```bibtex
@article{yang2023st,
  title={ST $^{2} $: Spatial-Temporal State Transformer for Crowd-aware Autonomous Navigation},
  author={Yang, Yuxiang and Jiang, Jiahao and Zhang, Jing and Huang, Jiye and Gao, Mingyu},
  journal={IEEE Robotics and Automation Letters},
  year={2023},
  publisher={IEEE}
}
```
