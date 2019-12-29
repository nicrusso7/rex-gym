# Rex-gym: OpenAI Gym environments for a quadruped robot
This repository contains different `Gym Environments` used to train Rex, the Rex URDF model, 
the learning agent and some scripts to start the training session and visualise the `Control Polices`.
## Rex: an open-source domestic robot
The goal of this project is to train an open-source 3D printed quadruped robot exploring 
`Reinforcement Learning` and `OpenAI Gym`. The aim is to let the robot learns domestic and generic tasks in the simulations and then 
successfully transfer the knowledge  (`Control Policies`) on the real robot without any 
other manual tuning.

This project is mostly inspired by the incredible works done by Boston Dynamics.
# Installation
Create a `Python 3.7` virtual environment, e.g. using `Anaconda`
```
conda create -n rex python=3.7 anaconda
conda activate rex
```
### PyPI package
Install the `rex-gym` package:
```
pip install rex_gym
```

### Install from source
You can also clone this repository and install it using `pip`. From the root of the project:
```
pip install .
```

# Run pre-trained agent simulation
To start a pre-trained agent:
```
python -m rex_gym.playground.rex_reactive_env_play
```
Check out the recorded video files under `/videos`.

# Start a new training simulation
To start a new training session:
```
python -m rex_gym.agents.scripts.train --config rex_reactive --logdir YOUR_LOG_DIR_PATH 
```
Where `YOUR_LOG_DIR_PATH` is the output policy path. 

Choose the `Gym Environment` setting the `--config` flag:

| Task      | Flag         |
|---------- | ------------ |
| Run       | rex_reactive |
| Walk      | rex_walking  |

## PPO Agent configuration
You may want to edit the PPO agent's default configuration, especially the number of parallel agents launched during 
the simulation.  

Edit the `num_agents` variable in the `agents/scripts/configs.py` script:

```
def default():
    """Default configuration for PPO."""
    # General
    ...
    num_agents = 20
```
Install rex_gym from source. This configuration will launch 20 agents (threads) in parallel to train your model.

# Robot platform
The robot used for this experiment is the [Spotmicro](https://www.thingiverse.com/thing:3445283) made by [Deok-yeon Kim](https://www.thingiverse.com/KDY0523/about).

[<img src="https://thingiverse-production-new.s3.amazonaws.com/assets/bf/af/74/db/83/complete_4.jpg">](https://www.thingiverse.com/thing:3445283)

I've printed the components using a Creality Ender3 3D printer, with PLA and TPU+ (this last one just for the foot 
cover).

The idea is to extend the basic robot adding components like a 3 joints robotic arm on the top of the rack and a 
Lidar sensor.

## Simulation model
Rex is a 12 joints robot with 3 motors (`Shoulder`, `Leg` and `Foot`) for each leg. 
The Rex `pose signal` (see ```rex_reactive_env.py```) sets the 12 motor angles that make Rex stands up.

The robot model was imported in `pyBullet` creating an [URDF file](rex_gym/util/pybullet_data/assets/urdf/rex.urdf). 

![rex bullet](rex_gym/util/images/rex.png)

# Tasks
This is the list of basic tasks I'd like to teach to Rex:

1. Locomotion - Run/Walk
2. Stand up - Falling recovery
3. Reach a specific point in a map
5. Grab an object

## Locomotion: Run
This task is about let Rex learns how to run on a straight line. 
### Gym Environment
There is a good number of papers on quadrupeds locomotion, most of them with sample code. Probably, the most complete collection 
of examples is the [Minitaur folder](https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs/minitaur) in the Bullet3 repository. 
For this task, I've edited the ```Minitaur Reactive Environment``` explained in the paper [Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/pdf/1804.10332.pdf).

#### Galloping gait - from scratch
In this very first experiment, I let the system learn from scratch: giving the feedback component large output bounds `[−0.5,0.5]` radians.
The `leg model` (see ```rex_reactive_env.py```) forces legs and foots movements (positive or negative direction, depending on the leg) influencing the learning 
score and time. In this first version, the `leg model` holds the Shoulder motors in the start position (0 degrees).  

As in the Minitaur example, I choose to use Proximal Policy Optimization (PPO). 

![](rex_gym/util/images/run.gif)

The emerged galloping gait shows the chassis tilled up and some unusual positions/movements (especially starting from the initial pose) during the locomotion. The `leg model` needs improvements. 

#### Galloping gait - bounded feedback
To improve the gait, in this second simulation, I've worked on the `leg model`:

![](rex_gym/util/images/leg_model-bounds.png) 

I set bounds for both `Leg` and `Foot` angles, keeping the `Shoulder` in the initial position.

![](rex_gym/util/images/galloping.gif)

The emerged gait now looks more clear.
#### Galloping gait - balanced feedback
Another test was made using a `balanced` feedback:

![](rex_gym/util/images/leg_model_improved.png) 

The Action Space dimension is equals to 4, assigning the same angle to both the front legs and a different one to the rear ones.
The very same was done for the foot angles. 

The simulation score is massively improved (about 10x) as the learning time while the emerged gait is very similar to the `bounded feedback` model. 
The Tensorflow score with this model, after ~500k attempts, is the same after ~4M attempts using any other models.
## Locomotion: Walk
This task is about let Rex learns how to walk on a straight line.
### Gym Environment
Starting from the [Minitaur Alternating Leg](https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/minitaur/envs/minitaur_alternating_legs_env.py)
example, I've used a sinusoidal signal as `leg_model` alternating the Rex legs during the locomotion. The feedback component has small bounds [-0.1,0.1] as in the original script, 
the sinusoidal function and the legs poses were edited. 

![](rex_gym/util/images/walk.gif)

# Credits
[Sim-to-Real: Learning Agile Locomotion For Quadruped Robots](https://arxiv.org/pdf/1804.10332.pdf) and all the related papers. Google Brain, Google X, Google DeepMind - Minitaur Ghost Robotics.

[Deok-yeon Kim](https://www.thingiverse.com/KDY0523/about) creator of SpotMini.

The great work in rendering the robot platform done by the [SpotMicroAI](https://github.com/FlorianWilk/SpotMicroAI) community.

