Model defintions for variations of the Probabilistic U-Net [Kohl et al. 2018](https://arxiv.org/abs/1806.05034). Repository is based upon [this implementation](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch), which you can check for further training details. Code includes the [PU-Net](https://arxiv.org/abs/1806.05034), [PU-Net+NF](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_8) and [SPU-NET](https://ieeexplore.ieee.org/abstract/document/10639444) model.

Note that models use bilinear upsampling by default for more consistent experiments and reproducibility at almost no cost of performance. Try out for yourself..

If appropriate, please cite:

```bibtex

@ARTICLE{10639444,
  author={Amaan Valiuddin, M. M. and Viviers, Christiaan G. A. and Van Sloun, Ruud J. G. and De With, Peter H. N. and Sommen, Fons van der},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Investigating and Improving Latent Density Segmentation Models for Aleatoric Uncertainty Quantification in Medical Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Uncertainty;Image segmentation;Probabilistic logic;Decoding;Training;Biomedical imaging;Annotations;Probabilistic Segmentation;Aleatoric Uncertainty;Latent Density Modeling},
  doi={10.1109/TMI.2024.3445999}}
