# Differentiable Scattering Delay Networks For Artificial Reverberation

Scattering delay networks (SDNs) provide a flexible and efficient framework for artificial reverberation and room acoustic modeling. We introduce differentiable SDNs, enabling gradient-based optimization of the model parameters to better approximate the acoustics of real-world environments. By formulating key parameters such as scattering matrices and absorption filters as differentiable functions, we employ gradient descent to optimize an SDN based on a target room impulse response. Our approach minimizes discrepancies in perceptually relevant acoustic features, such as energy decay and frequency-dependent reverberation times. Experimental results demonstrate that the learned SDN configurations significantly improve the accuracy of synthetic reverberation, highlighting the potential of data-driven room acoustic modeling.

## Training

To optimize an SDN model on the [HOMULA-RIR](https://github.com/polimi-ispl/homula-rir) data included in this repo, run the training script:

```console
foobar:~$ python train_homula.py --config config/homula-rir/your_config.yaml
```

> 💡 You can create custom configuration files under `config/` to adjust hyperparameters.

**Available config files:**
* `homula-rir/prior_fir0.yaml` trainable SDN with scalar wall absorption.
* `homula-rir/correction_fir0.yaml` trainable SDN with scalar wall absorption + distance correction.
* `homula-rir/prior_fir6.yaml` trainable SDN with FIR wall absorption filters.
* `homula-rir/correction_fir6.yaml` trainable SDN with FIR wall absorption filters + distance correction.

## Inference
You can produce the impulse response (IR) of an SDN model as follows:

```python
import torch

# Load a pre-trained SDN model
model = torch.load('path/to/model/checkpoint.pt')
model.eval()

# Instantiate a unit pulse
unit_pulse = torch.zeros(n_samples, device=model.device, dtype=model.dtype)
unit_pulse[0] = 1.

# Forward pass through the model
with torch.no_grad():
    ir = model(unit_pulse)

# Normalize to unit norm
ir /= ir.norm()
```

## Citation
If you use this repository in your research, please cite:

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
