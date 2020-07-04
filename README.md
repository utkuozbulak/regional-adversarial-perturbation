# Regional Adversarial Perturbation

This repository contains the implementation of the regional adversarial attack proposed in the "Regional Image Perturbation Reduces Lp Norms of Adversarial Examples While Maintaining Model-to-model Transferability" published in 2020 International Conference on Machine Learning, Workshop in Uncertainty & Robustness in Deep Learning ([Link to the paper](https://arxiv.org/abs/...)).


## General Information
We experimented with a simple and general method of localizing adversarial perturbation and used Cross-Entropy Sign (used by FGS, IFGS, and PGD) to generate adversarial examples with regional perturbation. Experimented regions are indicated below. 
<p align="center">
<img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/localization.png">
</p>

Some of the produced adversarial examples are provided below. We observed that regional perturbation reducing Lp, with p in {0, 2, \infty}, norms of adversarial perturbation while maintaining model-to-model transferability, thus posing a serious threat to defenses that claim robustness under certain Lp norms.

<p align="center">
<img src="https://raw.githubusercontent.com/utkuozbulak/cnn-gifs/master/adv_examples.png">
</p>

## Citation
If you find the code in this repository useful for your research, consider citing our paper.

    @inproceedings{ozbulak2020regional,
        title={Regional Image Perturbation Reduces Lp Norms of Adversarial Examples While Maintaining Model-to-model Transferability},
        author={Ozbulak, Utku and Van Messem, Arnout and De Neve, Wesley},
        booktitle={published in 2020 International Conference on Machine Learning, Workshop in Uncertainty and Robustness in Deep Learning},
        year={2020}
    }


## Requirements
```
python > 3.5
torch >= 0.4.0
torchvision >= 0.1.9
numpy >= 1.13.0
PIL >= 1.1.7
```
