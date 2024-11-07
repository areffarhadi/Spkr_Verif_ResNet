#! /usr/bin/env python3
import argparse
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from spk_veri_metric import SVevaluation
from dataset import WavDataset
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml


def extract_embeddings(model, val_dataloader, hparams, device, output_path):
    embd_dict = {}
    embd_stack = np.zeros([0, hparams['embd_dim']])  # Initialize empty embedding stack

    model.eval()
    with torch.no_grad():
        for j, (feat, utt) in enumerate(tqdm(val_dataloader)):
            embd = model(feat.to(device)).cpu().numpy()
            embd_dict[utt[0]] = embd  # Store embedding in dict with utterance ID
            embd_stack = np.concatenate((embd_stack, embd))  # Append embeddings

    # Save embeddings
    np.save(os.path.join(output_path, 'embd.npy'), embd_dict, allow_pickle=True)
    print(f"Embeddings saved to: {os.path.join(output_path, 'embd.npy')}")


def evaluate_embeddings(hparams, val_utt, output_path):
    # Load saved embeddings
    embd_dict = np.load(os.path.join(output_path, 'embd.npy'), allow_pickle=True).item()   #embd.npy
    
    # Stack embeddings into an array
    embd_stack = np.vstack([embd_dict[utt] for utt in val_utt])

    # Check if trials file exists and perform evaluation
    if os.path.exists('%s/trials' % hparams['val_name']):
        eer_cal = SVevaluation('%s/trials' % hparams['val_name'], val_utt)  #         eer_cal = SVevaluation('%s/trials' % hparams['val_name'], val_utt, ptar=[0.01])
        eer_cal.update_embd(embd_stack)
        # eer, cost, c_primary, eer_threshold, llr = eer_cal.eer_cost()
        # print(f"EER: {eer}, Cost: {cost}, C_primary: {c_primary}, Threshold: {eer_threshold}, LLR: {llr}")
        
        eer_cal.compute_llr_for_trials(output_file='LR_lang.txt')

        # eer, cost = eer_cal.eer_cost()
        # print(f"EER: {eer}, Cost: {cost}")
    else:
        print(f"Trials file not found in {hparams['val_name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASV evaluation")
    parser.add_argument('--yaml_path', required=True, type=str)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--mode', required=True, type=str, choices=['extract', 'evaluate'], help="Mode: 'extract' or 'evaluate'")

    args = parser.parse_args()

    # Load hyperparameters from YAML
    with open(args.yaml_path, 'r') as f:
        yaml_strings = f.read()
        hparams = load_hyperpyyaml(yaml_strings)

    # Prepare validation utterances and dataset
    val_utt = [line.split()[0] for line in open('%s/wav.scp' % hparams['val_name'])]

    # Load dataset
    val_dataset = WavDataset(
        [line.split() for line in open('%s/wav.scp' % hparams['val_name'])],
        norm_type=hparams['norm_type']
    )

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        shuffle=False,
        batch_size=1,
    )

    # Load pretrained model
    model = hparams['model']
    state_dict = torch.load(hparams['ckpt_path'], map_location=args.device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
    model = model.to(args.device)

    # Switch between 'extract' and 'evaluate' mode
    if args.mode == 'extract':
        extract_embeddings(model, val_dataloader, hparams, args.device, args.output_path)
    elif args.mode == 'evaluate':
        evaluate_embeddings(hparams, val_utt, args.output_path)


