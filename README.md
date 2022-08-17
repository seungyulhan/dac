# Diversity Actor Critic

This repository is an implementation of Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration (ICML 2021)
```
@article{han2020diversity,
  title={Diversity Actor-Critic: Sample-Aware Entropy Regularization for Sample-Efficient Exploration},
  author={Han, Seungyul and Sung, Youngchul},
  journal={arXiv preprint arXiv:2006.01419},
  year={2020}
}
```

## Dependencies

The implementation is based on [the source code](https://github.com/rail-berkeley/softlearning) of soft actor-critic [SAC](https://github.com/haarnoja/sac)

This implementation requires Anaconda / rllab / Mujoco / Tensorflow.

## Local Installation

1. Clone rllab [rllab](https://github.com/rll/rllab):

```
cd <install_path>
git clone https://github.com/rll/rllab.git
cd rllab
git checkout b3a28992eca103cab3cb58363dd7a4bb07f250a0
export PYTHONPATH=$<install_path>:${PYTHONPATH}
```

2. Install Mujoco 131 [Download](http://roboti.us/download.html) at rllab path:

```
mkdir -p /tmp/mujoco_tmp && cd /tmp/mujoco_tmp
wget -P . http://www.roboti.us/download/mjpro131_linux.zip
unzip mjpro131_linux.zip
mkdir <install_path>/rllab/vendor/mujoco
cp ./mjpro131/bin/libmujoco131.so <install_path>/rllab/vendor/mujoco
cp ./mjpro131/bin/libglfw.so.3 <install_path>/rllab/vendor/mujoco
cd ..
rm -rf /tmp/mujoco_tmp
```

3. Get Mujoco license key (mjkey.txt) [Download](https://web.archive.org/web/20220315114911/https://roboti.us/file/mjkey.txt) and copy it to rllab path:
```
cp <mujoco_key_folder>/mjkey.txt <install_path>/rllab/vendor/mujoco
export MUJOCO_PY_MJKEY_PATH="/home/user/rllab/vendor/mujoco/mjkey.txt"
export MUJOCO_PY_MJPRO_PATH="/home/user/rllab/vendor/mujoco/mjpro131"
```

4. Create conda environment and add path:
```
conda create -n dac python=3.6
export PATH="/home/<user_name>/anaconda3/envs/dac/bin:$PATH"
```

5. Install libraries and packages:
```
sudo apt-get update
sudo apt-get install python3-pip mpich libopenmpi-dev libgl-dev libglu-dev libxrandr-dev libxinerama-dev libxi-dev libxcursor-dev
conda activate mme

pip install numpy scipy path.py python-dateutil joblib==0.10.3 mako ipywidgets numba flask pygame h5py matplotlib mpi4py torchvision==0.1.6 pandas Pillow atari-py ipdb boto3 PyOpenGL nose2 pyzmq tqdm msgpack-python mujoco_py==0.5.7 cached_property line_profiler cloudpickle Cython redis git+https://github.com/Theano/Theano.git@adfe319ce6b781083d8dc3200fb4481b00853791#egg=Theano git+https://github.com/neocxi/Lasagne.git@484866cf8b38d878e92d521be445968531646bb8#egg=Lasagne plotly git+https://github.com/rll/rllab.git@b3a28992eca103cab3cb58363dd7a4bb07f250a0#egg=rllab gym==0.7.4 awscli pyglet jupyter progressbar2 tensorflow==1.4 numpy-stl==2.2.0 nibabel==2.1.0 pylru==1.0.9 hyperopt polling gtimer git+https://github.com/neocxi/prettytensor.git pyprind scikit-learn==0.20.0
```

## Training on Sparse-Rewarded Mujoco tasks

1. Fixed alpha

```
python -m examples.run_dac --env=half-cheetah --task sparse --alpha_adapt 0 --fixed_alpha 0.5 --gamma 0.99
```

2. Alpha-adaptation

```
python -m examples.run_dac --env=half-cheetah --task delayed --alpha_adapt 1 --ctrl_coef 1.0 --gamma 0.99
```
