import numpy as np
import torch
import torch.nn as nn

import os

import wandb
from utils import utils as utils


class CLIP(nn.Module):
    def __init__(self, text_input_dim, image_input_dim, latent_dim):
        super().__init__()

        self.text_input_dim = text_input_dim
        self.image_input_dim = image_input_dim
        self.latent_dim = latent_dim

        self.text_fc = nn.Linear(text_input_dim, latent_dim)
        self.text_af = nn.ReLU()
        self.text_fc2 = nn.Linear(latent_dim, latent_dim)
        self.image_fc = nn.Linear(image_input_dim, latent_dim)
        self.image_af = nn.ReLU()
        self.image_fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, text, image):
        # [batch_size, latent_dim]
        text_latent = self.text_fc2(self.text_af(self.text_fc(text)))
        image_latent = self.image_fc2(self.image_af(self.image_fc(image)))

        text_norms = torch.linalg.norm(text_latent, axis=1)
        image_norms = torch.linalg.norm(image_latent, axis=1)

        text_norms_repeated = text_norms.repeat(len(image), 1).T
        image_norms_repeated = image_norms.repeat(len(text), 1).T

        cosine_sim_unnormalised = text_latent @ image_latent.T
        cosine_sim_normalised = cosine_sim_unnormalised / text_norms_repeated / image_norms_repeated.T

        return cosine_sim_normalised


def evaluate(args, model, data):
    device = args.device

    correct = 0
    total = 0

    n_ways = args.num_ways

    model.eval()

    for i, batch in enumerate(data):
        batch_text = batch[1].to(device)
        batch_image = batch[0].to(device)
        batch_ids = batch[2]

        batch_size = batch_text.shape[0]

        shot_i = 0
        # TODO - Append leftovers to next batch
        while shot_i + n_ways < batch_size:
            shot_text = batch_text[shot_i].unsqueeze(0)
            shot_image = batch_image[shot_i:shot_i + n_ways]
            shot_ids = batch_ids[shot_i:shot_i + n_ways]

            with torch.no_grad():
                preds = model(shot_text, shot_image)

            if preds.argmax() == 0:
                correct += 1
            total += 1

            shot_i += n_ways

    return correct / total


def training_run(args, model, optimizer, train_loader, val_loader, n_epochs):
    device = args.device

    best_acc = evaluate(args, model, val_loader)
    best_epoch = 0
    print('init val_acc', best_acc)

    for epoch in range(n_epochs):
        model.train()
        model.zero_grad()

        for bid, batch in enumerate(train_loader):
            batch_text = batch[1].to(device)
            batch_image = batch[0].to(device)
            batch_ids = batch[2]

            # Discard repeated classes
            _, unique_idxs = torch.LongTensor(
                np.unique(batch_ids, return_index=True)).to(device)
            batch_text = batch_text[unique_idxs]
            batch_image = batch_image[unique_idxs]

            batch_size = batch_text.shape[0]

            optimizer.zero_grad()

            output = model(batch_text, batch_image)
            
            # Loss - Symmetric cross entropy
            labels = torch.arange(batch_size).to(device)
            loss_1 = nn.CrossEntropyLoss()(output, labels)
            loss_2 = nn.CrossEntropyLoss()(output.T, labels)

            loss = (loss_1 + loss_2) / 2.
            loss.backward()
            optimizer.step()

        # Val
        val_acc = evaluate(args, model, val_loader)
        print('epoch', epoch, 'val_acc', val_acc)
        wandb.log({'val/acc': val_acc}, step=epoch)
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch

        # save checkpoint
        checkpoint_dict = {
            "batch_idx": epoch,
            "state_dict": model.state_dict(),
            "best_loss": best_acc,
            "optimizer": optimizer.state_dict(),
            "args": vars(args)
        }
        utils.save_checkpoint(checkpoint_dict, is_best)

        # break after max iters or early stopping
        if (args.patience > 0 and epoch - best_epoch > args.patience):
            break

    # load best model    
    best_file = os.path.join(wandb.run.dir, "best.pth.tar")
    model, _ = utils.load_checkpoint(model, optimizer, args.device, best_file)

    return model
