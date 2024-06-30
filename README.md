## Educational examples for implementations of various diffusion models

Each implmenentation of diffusion models is self-contained. 

Python module requirements:
```
pip install pytorch
pip install tqdm
pip install torchvision
```

## How to use
* Denoising Diffusion Probabilistic Models (DDPM): [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
``` bash
python3 ddpm.py
```

* Explicit Conditional DDPM
``` bash
python3 conditional_ddpm.py
```

* Classifier Guided DDPM: [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
``` bash
python3 guided_ddpm.py
```

* Classifier Free DDPM: [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
``` bash
python3 classifier_free_ddpm.py
```
