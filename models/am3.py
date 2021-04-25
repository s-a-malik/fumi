"""Model classes and training loops for AM3 in Pytorch.
"""

import wandb
import torch
import torch.nn as nn

from transformers import BertModel
from tqdm.autonotebook import tqdm

import utils
from models.common import WordEmbedding


class AM3(nn.Module):
    def __init__(self, im_encoder, im_emb_dim, text_encoder, text_emb_dim=300, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False, dictionary=None, pooling_strat="mean"):
        super(AM3, self).__init__()
        self.im_emb_dim = im_emb_dim            # image embedding size
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim        # only applicable if precomputed
        self.text_hid_dim = text_hid_dim        # AM3 uses 300
        self.prototype_dim = prototype_dim      # AM3 uses 512 (resnet)
        self.dropout = dropout                  # AM3 uses 0.7 or 0.9 depending on dataset
        self.fine_tune = fine_tune
        self.dictionary = dictionary            # for word embeddings
        self.pooling_strat = pooling_strat

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
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_emb_dim = self.text_encoder.config.hidden_size
        elif self.text_encoder_type == "precomputed":
            self.text_encoder = nn.Identity()
        elif self.text_encoder_type == "w2v" or self.text_encoder_type == "glove":
            # load pretrained word embeddings as weights
            self.text_encoder = WordEmbedding(self.text_encoder_type, self.pooling_strat, self.dictionary)
            self.text_emb_dim = self.text_encoder.embedding_dim
        elif self.text_encoder_type == "RNN":
            # TODO RNN implementation
            self.text_encoder = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        elif self.text_encoder_type == "rand":
            self.text_encoder = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        else:
            raise NameError(f"{text_encoder} not allowed as text encoder")

        # fine tune set up
        if not self.fine_tune:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # text to prototype neural net
        self.g = nn.Sequential(
            nn.Linear(self.text_emb_dim, self.text_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.text_hid_dim, self.prototype_dim)
        )

        # text prototype to lamda neural net
        self.h = nn.Sequential(
            nn.Linear(self.prototype_dim, self.text_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.text_hid_dim, 1)
        )

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
        if self.text_encoder_type == "BERT":
            idx, text, attn_mask, im = inputs
        else:
            idx, text, im = inputs

        # process
        im_embeddings = self.image_encoder(im)      # (b x N*K x 512)  
        if im_only:
            return im_embeddings
        else:
            B, NK, seq_len = text.shape
            if self.text_encoder_type == "BERT":
                # need to reshape batch for BERT input
                bert_output = self.text_encoder(text.view(-1, seq_len), attention_mask=attn_mask.view(-1, seq_len))
                # get [CLS] token
                text_encoding = bert_output[1].view(B, NK, -1)        # (b x N*K x 768)
            elif self.text_encoder_type == "rand":
                # get a random tensor as the embedding
                # text_encoding = 2*torch.rand(B, NK, self.text_emb_dim) - 1
                pass
            else:
                text_encoding = self.text_encoder(text)
            
            if self.text_encoder_type == "rand":
                text_embeddings = 2*torch.rand(B, NK, self.prototype_dim) - 1
            else:
                text_embeddings = self.g(text_encoding)   # (b x N*K x 512)
            
            lamda = torch.sigmoid(self.h(text_embeddings))  # (b x N*K x 1)
            return im_embeddings, text_embeddings, lamda

    def evaluate(self, batch, optimizer, scheduler, num_ways, device, task="train"):
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
        train_im_embeddings, train_text_embeddings, train_lamda = self(train_inputs)
        avg_lamda = torch.mean(train_lamda)

        # query set
        test_inputs, test_targets = batch['test']
        test_inputs = [x.to(device) for x in test_inputs]
        test_targets = test_targets.to(device)
        test_im_embeddings = self(test_inputs, im_only=True)    # only get image prototype

        # TODO try using lambda = 0 or 1 
        # train_lamda = torch.ones_like(train_lamda)
        # train_lamda = torch.zeros_like(train_lamda)

        # construct prototypes
        prototypes = utils.get_prototypes(
            train_im_embeddings,
            train_text_embeddings,
            train_lamda,
            train_targets,
            num_ways)
        
        # this is cross entropy on euclidean distance between prototypes and embeddings
        loss = utils.prototypical_loss(prototypes, test_im_embeddings, test_targets)

        if task == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        with torch.no_grad():
            preds, acc = utils.get_preds(prototypes, test_im_embeddings, test_targets)
        
        if task == "test":
            test_idx = test_inputs[0]
            train_idx = train_inputs[0]
            return loss.detach().cpu().numpy(), acc, avg_lamda.detach().cpu().numpy(), preds, test_targets.detach().cpu().numpy(), test_idx.detach().cpu().numpy(), train_idx.detach().cpu().numpy(), train_lamda.squeeze(-1).detach().cpu().numpy()
        else:
            return loss.detach().cpu().numpy(), acc, avg_lamda.detach().cpu().numpy()


def training_run(args, model, optimizer, train_loader, val_loader, max_test_batches):
    """Run training loop
    Returns:
    - model (nn.Module): trained model
    """
    # get best val loss
    best_loss, best_acc, _, _, _, _, _, _ = test_loop(
        args, model, val_loader, max_test_batches)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_batch_idx = 0

    # check if scheduled
    if type(optimizer) == tuple:
        opt, scheduler = optimizer
    else:
        opt = optimizer
        scheduler = None

    # use try, except to be able to stop partway through training
    # TODO this doesn't work with wandb for some reason. Might be good? no eval on test set.
    # can always call --evaluate later with saved checkpoint
    try:
        # Training loop
        # do in epochs with a max_num_batches instead?
        for batch_idx, batch in enumerate(train_loader):
            train_loss, train_acc, train_lamda = model.evaluate(
                batch=batch,
                optimizer=opt,
                scheduler=scheduler,
                num_ways=args.num_ways,
                device=args.device,
                task="train")

            # log
            # TODO track lr etc as well if using scheduler
            wandb.log({"train/acc": train_acc,
                       "train/loss": train_loss,
                       "train/avg_lamda": train_lamda,
                       "num_episodes": (batch_idx+1)*args.batch_size}, step=batch_idx)

            # eval on validation set periodically
            if batch_idx % args.eval_freq == 0:
                # evaluate on val set
                val_loss, val_acc, val_lamda, _, _, _, _, _ = test_loop(
                    args, model, val_loader, max_test_batches)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_batch_idx = batch_idx
                wandb.log({"val/acc": val_acc,
                           "val/loss": val_loss,
                           "val/avg_lamda": val_lamda}, step=batch_idx)
                # TODO F1/prec/recall etc.?

                # save checkpoint
                checkpoint_dict = {
                    "batch_idx": batch_idx,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": opt.state_dict(),
                    "args": vars(args)
                }
                utils.save_checkpoint(checkpoint_dict, is_best)

                print(f"\nBatch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}, train/avg_lamda: {train_lamda}"
                      f"\nval/loss: {val_loss}, val/acc: {val_acc}, val/avg_lamda: {val_lamda}")

            # break after max iters or early stopping
            if (batch_idx > args.epochs - 1) or (batch_idx - best_batch_idx > args.patience):
                break
    except KeyboardInterrupt:
        pass

    return model


def test_loop(args, model, test_dataloader, max_num_batches):
    """Evaluate model on val/test set.
    Test on 1000 randomly sampled tasks, each with 100 query samples (as in AM3)
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_loss (float): average test loss per task
    """
    # TODO add more test metrics + example outputs?
    # TODO need to fix number of tasks/episodes etc. depending on batch, num_ways etc.

    avg_test_acc = utils.AverageMeter()
    avg_test_loss = utils.AverageMeter()
    test_preds = []
    test_trues = []
    query_idx = []
    support_idx = []
    support_lamdas = []
    avg_lamda = utils.AverageMeter()

    for batch_idx, batch in enumerate(tqdm(test_dataloader, total=max_num_batches, position=0, leave=True)):
        with torch.no_grad():
            test_loss, test_acc, lamda, preds, trues, query, support, support_lamda = model.evaluate(
                batch=batch,
                optimizer=None,
                scheduler=None,
                num_ways=args.num_ways,
                device=args.device,
                task="test")

        avg_test_acc.update(test_acc)
        avg_test_loss.update(test_loss)
        avg_lamda.update(lamda)
        test_preds += preds.tolist()
        test_trues += trues.tolist()
        query_idx += query.tolist()
        support_idx += support.tolist()
        support_lamdas += support_lamda.tolist()

        if batch_idx > max_num_batches - 1:
            break

    return avg_test_loss.avg, avg_test_acc.avg, avg_lamda.avg, test_preds, test_trues, query_idx, support_idx, support_lamdas


if __name__ == "__main__":

    model = AM3(im_encoder="precomputed", im_emb_dim=512, text_encoder="BERT", text_emb_dim=768, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False)
    print(model)
    N = 5
    K = 2
    B = 5
    idx = torch.ones(B, N*K)
    text = torch.ones(B, N*K, 128, dtype=torch.int64)
    im = torch.ones(B, N*K, 512)   
    targets = 2*torch.ones(B, N*K, dtype=torch.int64)
    inputs = (idx, text, im)
    im_embed, text_embed, lamdas = model(inputs)
    print("output shapes (im, text, lamda)", im_embed.shape, text_embed.shape, lamdas.shape)
    prototypes = utils.get_prototypes(im_embed, text_embed, lamdas, targets, N)
    print("prototypes", prototypes.shape)
    loss = utils.prototypical_loss(prototypes, im_embed, targets)   # test on train set
    print("loss", loss)
