# Differentiable Scattering Delay Networks For Artificial Reverberation

Scattering delay networks (SDNs) provide a flexible and efficient framework for artificial reverberation and room acoustic modeling. We introduce a differentiable SDN, enabling gradient-based optimization of its parameters to better approximate the acoustics of real-world environments. By formulating key parameters such as scattering matrices and absorption filters as differentiable functions, we employ gradient descent to optimize an SDN based on a target room impulse response. Our approach minimizes discrepancies in perceptually relevant acoustic features, such as energy decay and frequency-dependent reverberation times. Experimental results demonstrate that the learned SDN configurations significantly improve the accuracy of synthetic reverberation, highlighting the potential of data-driven room acoustic modeling.

## Citation
If you find this repo useful, please cite
```
@inproceedings{mezza2025differentiable,
    author = {Mezza, Alessandro Ilic and Giampiccolo, Riccardo and De Sena, Enzo and Bernardini, Alberto},
    title = {Differentiable Scattering Delay Networks For Artificial Reverberation},
    booktitle = {Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25)},
    address = {Ancona, Italy},
    month = {sep}, 
    year = {2025}
}
```
