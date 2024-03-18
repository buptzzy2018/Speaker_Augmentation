# sunine-dev: development version of [sunine](https://gitlab.com/csltstu/sunine).

## Quick installation
1. Clone this repo

```base
git clone https://gitlab.com/wangth2001/sunine-dev.git
```

2. Create conda env and install the requirements

```base
conda create -n sunine python=3.9
conda activate sunine
conda install pytorch==1.12.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```


## Methodologies

### Data processing
+ On-the-fly acoustic feature extraction: PreEmphasis and MelSpectrogram.
+ On-the-fly data augmentation: additive noises on [MUSAN](http://www.openslr.org/17/) and reverberation on [RIR](http://www.openslr.org/28/).

### Backbone
+ [x] [TDNN](https://ieeexplore.ieee.org/abstract/document/8461375)
+ [x] [ResNet34](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
+ [x] [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
+ [x] [CAMPPlus](https://arxiv.org/abs/2303.00332)

### Pooling
+ [x] [Temporal Average Pooling](https://arxiv.org/abs/1903.12058)
+ [x] [Temporal Statistics Pooling](http://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
+ [x] [Self-Attentive Pooling](https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf)
+ [x] [Attentive Statistics Pooling](https://arxiv.org/abs/1803.10963)

### Loss Function
+ [x] [Softmax](https://ieeexplore.ieee.org/abstract/document/8461375)
+ [x] [AM-Softmax](https://arxiv.org/abs/1801.05599)
+ [x] [AAM-Softmax](https://arxiv.org/abs/1801.07698)
+ [x] [ARM-Softmax](https://arxiv.org/abs/2110.09116)
+ [x] [SphereFace2](https://arxiv.org/abs/2108.01513)

### Training Strategy
+ [x] Learning rate warm-up
+ [x] Margin Scheduler
+ [x] Large margin fine-tuning

### Backend
+ [x] Cosine
+ [x] [PLDA](https://link.springer.com/chapter/10.1007/11744085_41)
+ [x] Score Normalization: [S-Norm](https://www.isca-speech.org/archive/odyssey_2010/kenny10_odyssey.html), [AS-Norm](https://www.isca-speech.org/archive_v0/archive_papers/interspeech_2011/i11_2365.pdf)

### Metric
+ [x] Calibration: [Cllr, minCllr](https://www.sciencedirect.com/science/article/pii/S0885230805000483)
+ [x] EER
+ [x] minDCF
+ [x] Top-K ACC
