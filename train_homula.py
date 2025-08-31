import yaml
import torch
import pickle
import argparse
from pathlib import Path
from utils import seed_everything, load_homula_rir
from sdn import SDN
import losses


def main(args):
    # Set seeds for reproducibility
    seed_everything(42)

    # Load room configuration
    with open(args.room) as stream:
        try:
            room = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Load training configuration
    with open(args.config) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    dtype = torch.float32
    device = args.device
    sr = config['sr']
    n_epochs = config['training']['n_epochs']

    # Factory kwargs used to move tensors/models to device and dtype consistently
    factory_kwargs = {"device": device, "dtype": dtype}

    # Load RIR from HOMULA-RIR
    true_rir = load_homula_rir(rir_path=room['rir']['path'], ula_index=room['rir']['ula_index'], sr=sr, trim=True)

    # Move RIR to device
    true_rir = true_rir.to(**factory_kwargs)

    # Instantiate the input unit pulse
    x = torch.zeros(true_rir.shape[-1]).to(**factory_kwargs)
    x[0] = 1.

    # Instantiate the SDN model
    sdn = SDN(
        room_dim=room['room_dim'],
        src_pos=room['src_pos'],
        mic_pos=room['mic_pos'],
        sr=sr,
        c=config['c'],
        N=config['sdn']['N'],
        junction_type=config['sdn']['junction_type'],
        delay_buffer_len=config['sdn']['delay_buffer_len'],
        train_distances=config['sdn']['train_distances'],
        max_distance_correction=config['sdn']['max_distance_correction'],
        distance_scaling=config['sdn']['distance_scaling'],
        fir_order=config['sdn']['fir_order'],
        alpha=config['sdn']['alpha'],
        **factory_kwargs
    )

    # Define (weighted) losses
    sdn_loss_functions = [  # (loss_name, loss_fn, lambda),
        ('EDC', losses.EDCLoss(), config['training']['lambda_edc']),
        ('EDR', losses.MelEDRLogLoss(sr=sr), config['training']['lambda_edr']),
        ('EDP', losses.EDPLoss(sr=sr), config['training']['lambda_edp']),
    ]

    # Define the optimizer
    optimizer = torch.optim.Adam(sdn.parameters(), lr=config['training']['learning_rate'])

    # Instantiate the data structure for storing loss values
    loss_history = {k[0]: [] for k in sdn_loss_functions}

    # Start training
    for epoch in range(n_epochs):
        print(f'Epoch: {epoch + 1}/{n_epochs}')
        # Reset the optimizer's gradients before backpropagation
        optimizer.zero_grad()

        # Forward pass: estimate the impulse response of the SDN model
        pred_rir = sdn(x)

        # Normalize the prediction to unit norm
        pred_rir = pred_rir / pred_rir.norm()

        # Accumulate loss
        loss = 0.0
        for (loss_name, loss_fn, lmbda) in sdn_loss_functions:
            loss_term = loss_fn(pred_rir, true_rir)
            loss_history[loss_name].append(loss_term.item())
            loss += lmbda * loss_term

        # Save model checkpoint for this epoch
        torch.save(sdn, save_dir.joinpath(f'sdn_epoch_{epoch}.pth'))

        # Save current loss history to file (so progress isn’t lost if training stops)
        with open(save_dir.joinpath('loss_history.pickle'), 'wb') as stream:
            pickle.dump(loss_history, stream, protocol=pickle.HIGHEST_PROTOCOL)
        print('Model saved.')

        # Update model parameters via backpropagation
        loss.backward()
        optimizer.step()

    # Save model checkpoint for this epoch
    torch.save(sdn, save_dir.joinpath(f'sdn_epoch_{n_epochs}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-r', '--room', default='config/rooms/schiavoni_room.yaml')
    parser.add_argument('-d', '--device', default='cuda:0')
    args = parser.parse_args()
    main(args)
