# Graph-Propagation-based Kinematic Algorithm for In-pipe Truss Structure Robots

Open source code for RA-L paper [Graph-Propagation-based Kinematic Algorithm for In-pipe Truss Structure Robots](https://ieeexplore.ieee.org/abstract/document/10494897).

[Yu Chen](https://neuling-jpg.github.io/yuchen.github.io/), [Jinyun Xu](https://www.ri.cmu.edu/ri-people/jinyun-xu/), [Yilin Cai](https://missinglight.github.io/), [Shuo Yang](https://shuoyangrobotics.github.io/), [Ben Brown](https://www.cs.cmu.edu/~hbb/), [Fujun Ruan](https://fujunruan.com/), [Yizhu Gu](https://www.ri.cmu.edu/ri-people/yizhu-gu/), [Howie Choset](https://www.cs.cmu.edu/~choset/), [Lu Li](https://www.ri.cmu.edu/ri-people/lu-li/)

https://github.com/Neuling-jpg/Earthworm/assets/65380456/90371f0b-084b-4821-9c71-93bc2e73ea83

## Setup

This project relies on a basic environment: Python3 + numpy + [vpython](https://pypi.org/project/vpython/), where `vpython` is a lightweight package for visualization.

```bash
git clone https://github.com/Neuling-jpg/Earthworm.git
pip install numpy
pip install vpython
```

## Run examples

### End effector path following

```bash
python ee_path_follow.py --use_propagation 1 --solver fmin_bfgs --n_nodes 14 --freeze_num 0 --path_name sin --visualize 1
python ee_path_follow.py --use_propagation 1 --solver fmin_bfgs --n_nodes 14 --freeze_num 0 --path_name circle --visualize 1
python ee_path_follow.py --use_propagation 1 --solver fmin_bfgs --n_nodes 14 --freeze_num 0 --path_name polynomial --visualize 1
python ee_path_follow.py --use_propagation 1 --solver fmin_bfgs --n_nodes 14 --freeze_num 0 --path_name cmu --visualize 1
```

### Straight pipe crawling

```bash
python crawl_straight_pipe.py --use_propagation 1 --solver fmin_bfgs --freeze_num 0 --num_steps 50 --visualize 1
```

### Pipe bends crawling

```bash
python crawl_torus_pipe.py --use_propagation 1 --solver fmin_bfgs --freeze_num 0 --num_steps 50 --visualize 1
```