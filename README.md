# awesome-gans

### Setup
Installs pip requirements and expands dataset
```bash
git clone https://github.com/stevenctl/awesome-gans
virtualenv venv -p=$(which python3)
source venv/bin/activate
./setup.sh
```


### CycleGAN

A broken-down version of the Tensorflow CycleGAN tutorial found here:
https://www.tensorflow.org/beta/tutorials/generative/cyclegan

It also contains a very small dataset of images of cars to train on.

#### Running it:

**in iPython:** `ipython train_cyclegan.py`

**in Notebook:** `import trayin_cyclegan`


