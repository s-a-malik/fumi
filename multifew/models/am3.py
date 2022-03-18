"""Model classes and training loops for AM3 in Pytorch.
"""

import os
import wandb
import torch
import torch.nn as nn

from transformers import BertModel
from tqdm.autonotebook import tqdm

from utils.average_meter import AverageMeter
from utils import utils as utils
from .common import WordEmbedding, RNN, RnnHid


class AM3(nn.Module):
    def __init__(self,
                 im_encoder,
                 im_emb_dim,
                 text_encoder,
                 text_emb_dim=300,
                 text_hid_dim=300,
                 prototype_dim=512,
                 dropout=0.7,
                 fine_tune=False,
                 dictionary=None,
                 pooling_strat="mean",
                 lamda_fixed=None):
        super(AM3, self).__init__()
        self.im_emb_dim = im_emb_dim  # image embedding size
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim  # only applicable if precomputed or RNN hid dim.
        self.text_hid_dim = text_hid_dim  # AM3 uses 300
        self.prototype_dim = prototype_dim  # AM3 uses 512 (resnet)
        self.dropout = dropout  # AM3 uses 0.7 or 0.9 depending on dataset
        self.fine_tune = fine_tune
        self.dictionary = dictionary  # for word embeddings
        self.pooling_strat = pooling_strat
        self.lamda_fixed = lamda_fixed

        if im_encoder == "precomputed":
            # if using precomputed embeddings
            self.image_encoder = nn.Linear(self.im_emb_dim, self.prototype_dim)
        elif im_encoder == "resnet":
            # TODO image encoder if raw images
            self.image_encoder = nn.Linear(self.im_emb_dim, self.prototype_dim)
        else:
            raise NameError(f"{im_encoder} not allowed as image encoder")

        if self.text_encoder_type == "BERT":
            # TODO be able to use any hf bert model (requires correct tokenisation)
            # self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            # self.text_emb_dim = self.text_encoder.config.hidden_size
            self.text_encoder = nn.Identity()

        elif self.text_encoder_type == "precomputed":
            self.text_encoder = nn.Identity()
        elif self.text_encoder_type == "w2v" or self.text_encoder_type == "glove":
            # load pretrained word embeddings as weights
            self.text_encoder = WordEmbedding(self.text_encoder_type,
                                              self.pooling_strat,
                                              self.dictionary)
            self.text_emb_dim = self.text_encoder.embedding_dim
        elif self.text_encoder_type == "RNN":
            # TODO optional embedding type
            self.text_encoder = RNN("glove", self.pooling_strat,
                                    self.dictionary, self.text_emb_dim)
        elif self.text_encoder_type == "RNNhid":
            self.text_encoder = RnnHid("glove", self.pooling_strat,
                                    self.dictionary, self.text_emb_dim)
        elif self.text_encoder_type == "rand":
            self.text_encoder = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        else:
            raise NameError(f"{text_encoder} not allowed as text encoder")

        # fine tune set up
        if not self.fine_tune:
            print("Not fine tuning embeddings")
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # text to prototype neural net
        self.g = nn.Sequential(
            nn.Linear(self.text_emb_dim, self.text_hid_dim), nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.text_hid_dim, self.prototype_dim))

        # text prototype to lamda neural net
        self.h = nn.Sequential(
            nn.Linear(self.prototype_dim, self.text_hid_dim), nn.ReLU(),
            nn.Dropout(p=self.dropout), nn.Linear(self.text_hid_dim, 1))

    def forward(self, inputs, im_only=False):
        """
        Params:
        - inputs (tuple): 
            - idx:  image ids
            - text: padded tokenised sequence. (tuple for BERT of input_ids and attn_mask). 
            - attn_mask: attention mask for bert (only if BERT)
            - im:   precomputed image embeddings
        - im_only (bool): flag to only use image input (for query set)

        Returns:
        - im_embeddings (torch.FloatTensor): image in prototype space (b, NxK, emb_dim)
        - (if not im_only) text_embeddings (torch.FloatTensor): text in prototype space (b, NxK, emb_dim)
        """
        # unpack input
        # if self.text_encoder_type == "BERT":
        #     idx, text, attn_mask, im = inputs
        # else:
        #     idx, text, im = inputs
        
        # precomputed bert is the same
        idx, text, im = inputs

        # process
        im_embeddings = self.image_encoder(im)  # (b x N*K x 512)
        if im_only:
            return im_embeddings
        else:
            B, NK, seq_len = text.shape
            # if self.text_encoder_type == "BERT":
            #     # need to reshape batch for BERT input
            #     bert_output = self.text_encoder(text.view(-1, seq_len),
            #                                     attention_mask=attn_mask.view(
            #                                         -1, seq_len))
            #     # get [CLS] token
            #     text_encoding = bert_output[1].view(B, NK,
            #                                         -1)  # (b x N*K x 768)
            # elif self.text_encoder_type == "rand":
            #     # get a random tensor as the embedding
            #     # text_encoding = 2*torch.rand(B, NK, self.text_emb_dim) - 1
            #     pass
            # else:
            
            # bert precomputed
            text_encoding = self.text_encoder(
                text)  # (B, N*K, text_emb_dim)

            if self.text_encoder_type == "rand":
                text_embeddings = 2 * torch.rand(
                    size=(B, NK, self.prototype_dim),
                    device=im_embeddings.device) - 1
            else:
                text_embeddings = self.g(text_encoding)  # (b x N*K x 512)

            lamda = torch.sigmoid(self.h(text_embeddings))  # (b x N*K x 1)
            return im_embeddings, text_embeddings, lamda

    def evaluate(self,
                 batch,
                 optimizer,
                 scheduler,
                 num_ways,
                 device,
                 task="train"):
        """Run one episode through model
        Params:
        - batch (dict): meta-batch of tasks
        - optimizer (nn.optim): optimizer tied to model weights. 
        - scheduler: learning rate scheduler
        - num_ways (int): number 'N' of classes to choose from.
        - device (torch.device): cuda or cpu
        - task (str): train, val, test
        Returns:
        - loss: prototypical loss
        - acc: accuracy on query set
        - avg_lamda: average lamda for prototypes in batch
        - in addtion, if test:
            - preds: list of class predictions
            - test_targets: list of true classes
            - test_idx: list of query set image ids
            - train_idx: list of support set image ids
            - train_lamda: list of lamdas for each image in support set
        """
        if task == "train":
            self.train()
        else:
            self.eval()

        # support set
        train_inputs, train_targets = batch['train']
        train_inputs = [x.to(device) for x in train_inputs]
        train_targets = train_targets.to(device)
        train_im_embeddings, train_text_embeddings, train_lamda = self(
            train_inputs)

        # query set
        test_inputs, test_targets = batch['test']
        test_inputs = [x.to(device) for x in test_inputs]
        test_targets = test_targets.to(device)
        test_im_embeddings = self(test_inputs,
                                  im_only=True)  # only get image prototype

        # lambda set to 0 or 1
        if self.lamda_fixed == 0:
            train_lamda = torch.zeros_like(train_lamda).to(device)
        elif self.lamda_fixed == 1:
            train_lamda = torch.ones_like(train_lamda).to(device)
        else:
            train_lamda = train_lamda

        # construct prototypes
        prototypes = utils.get_prototypes(train_im_embeddings,
                                          train_text_embeddings, train_lamda,
                                          train_targets, num_ways)

        # this is cross entropy on euclidean distance between prototypes and embeddings
        loss = utils.prototypical_loss(prototypes, test_im_embeddings,
                                       test_targets)
        avg_lamda = torch.mean(train_lamda)

        if task == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            preds, acc, f1, prec, rec = utils.get_preds(
                prototypes, test_im_embeddings, test_targets)

        if task == "test":
            test_idx = test_inputs[0]
            train_idx = train_inputs[0]
            return loss.detach().cpu().numpy(
            ), acc, f1, prec, rec, avg_lamda.detach().cpu().numpy(
            ), preds, test_targets, test_idx.detach().cpu().numpy(
            ), train_idx.detach().cpu().numpy(), train_lamda.squeeze(
                -1).detach().cpu().numpy()
        else:
            return loss.detach().cpu().numpy(
            ), acc, f1, prec, rec, avg_lamda.detach().cpu().numpy()


def training_run(args, model, optimizer, train_loader, val_loader,
                 max_test_batches):
    """Run training loop
    Returns:
    - model (nn.Module): trained model
    """
    # get best val loss
    best_loss, best_acc, _, _, _, _, _, _, _, _, _ = test_loop(
        args, model, val_loader, max_test_batches)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_batch_idx = 0

    # check if scheduled
    if type(optimizer) == tuple:
        opt, scheduler = optimizer
    else:
        opt = optimizer
        scheduler = None

    try:
        # Training loop
        # do in epochs with a max_num_batches instead?
        for batch_idx, batch in enumerate(train_loader):
            train_loss, train_acc, train_f1, train_prec, train_rec, train_lamda = model.evaluate(
                batch=batch,
                optimizer=opt,
                scheduler=scheduler,
                num_ways=args.num_ways,
                device=args.device,
                task="train")

            # log
            # TODO track lr etc as well if using scheduler
            wandb.log(
                {
                    "train/acc": train_acc,
                    "train/f1": train_f1,
                    "train/prec": train_prec,
                    "train/rec": train_rec,
                    "train/loss": train_loss,
                    "train/avg_lamda": train_lamda,
                    "num_episodes": (batch_idx + 1) * args.batch_size
                },
                step=batch_idx)

            # eval on validation set periodically
            if batch_idx % args.eval_freq == 0:
                # evaluate on val set
                val_loss, val_acc, val_f1, val_prec, val_rec, val_lamda, _, _, _, _, _ = test_loop(
                    args, model, val_loader, max_test_batches)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_batch_idx = batch_idx
                wandb.log(
                    {
                        "val/acc": val_acc,
                        "val/f1": val_f1,
                        "val/prec": val_prec,
                        "val/rec": val_rec,
                        "val/loss": val_loss,
                        "val/avg_lamda": val_lamda
                    },
                    step=batch_idx)

                # save checkpoint
                checkpoint_dict = {
                    "batch_idx": batch_idx,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": opt.state_dict(),
                    "args": vars(args)
                }
                utils.save_checkpoint(checkpoint_dict, is_best)

                print(
                    f"\nBatch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}, train/avg_lamda: {train_lamda}"
                    f"\nval/loss: {val_loss}, val/acc: {val_acc}, val/avg_lamda: {val_lamda}"
                )

            # break after max iters or early stopping
            if (batch_idx > args.epochs - 1) or (args.patience > 0 and batch_idx - best_batch_idx >
                                                 args.patience):
                break
    except KeyboardInterrupt:
        pass

    # load best model    
    best_file = os.path.join(wandb.run.dir, "best.pth.tar")
    model, _ = utils.load_checkpoint(model, opt, args.device, best_file)

    return model


def test_loop(args, model, test_dataloader, max_num_batches):
    """Evaluate model on val/test set.
    Test on 1000 randomly sampled tasks, each with 100 query samples (as in AM3)
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_f1 (float): average test f1 per task
    - avg_test_prec (float): average test prec per task
    - avg_test_rec (float): average test rec per task
    - avg_test_loss (float): average test loss per task
    - avg_test_lambda (float): average test lambda per task
    - test_preds
    - test_trues
    - query_idx
    - support_idx
    - support_lamdas
    """
    # TODO need to fix number of tasks/episodes etc. depending on batch, num_ways etc.

    avg_test_acc = AverageMeter()
    avg_test_f1 = AverageMeter()
    avg_test_prec = AverageMeter()
    avg_test_rec = AverageMeter()
    avg_test_loss = AverageMeter()
    test_preds = []
    test_trues = []
    query_idx = []
    support_idx = []
    support_lamdas = []
    avg_lamda = AverageMeter()

    for batch_idx, batch in enumerate(
            tqdm(test_dataloader,
                 total=max_num_batches,
                 position=0,
                 leave=True)):
        with torch.no_grad():
            test_loss, test_acc, test_f1, test_prec, test_rec, lamda, preds, trues, query, support, support_lamda = model.evaluate(
                batch=batch,
                optimizer=None,
                scheduler=None,
                num_ways=args.num_ways,
                device=args.device,
                task="test")

        avg_test_acc.update(test_acc)
        avg_test_f1.update(test_f1)
        avg_test_prec.update(test_prec)
        avg_test_rec.update(test_rec)
        avg_test_loss.update(test_loss)

        avg_lamda.update(lamda)
        test_preds += preds.tolist()
        test_trues += trues.tolist()
        query_idx += query.tolist()
        support_idx += support.tolist()
        support_lamdas += support_lamda.tolist()

        if batch_idx > max_num_batches - 1:
            break

    return avg_test_loss.avg, avg_test_acc.avg, avg_test_f1.avg, avg_test_prec.avg, avg_test_rec.avg, avg_lamda.avg, test_preds, test_trues, query_idx, support_idx, support_lamdas


if __name__ == "__main__":
    import os, sys
    currentdir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(currentdir)
    sys.path.append(parentdir)

    model = AM3(im_encoder="precomputed",
                im_emb_dim=512,
                text_encoder="rand",
                text_emb_dim=768,
                text_hid_dim=300,
                prototype_dim=512,
                dropout=0.7,
                fine_tune=False)
    print(model)
    N = 5
    K = 2
    B = 5
    idx = torch.ones(B, N * K)
    text = torch.ones(B, N * K, 128, dtype=torch.int64)
    im = torch.ones(B, N * K, 512)
    targets = 2 * torch.ones(B, N * K, dtype=torch.int64)
    inputs = (idx, text, im)
    im_embed, text_embed, lamdas = model(inputs)
    print("output shapes (im, text, lamda)", im_embed.shape, text_embed.shape,
          lamdas.shape)
    prototypes = utils.get_prototypes(im_embed, text_embed, lamdas, targets, N)
    print("prototypes", prototypes.shape)
    loss = utils.prototypical_loss(prototypes, im_embed,
                                   targets)  # test on train set
    print("loss", loss)
    targets[:, 1] = 1
    preds, acc, f1, prec, rec = utils.get_preds(prototypes, im_embed, targets)
    print(preds)
