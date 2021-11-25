import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from transformers import BertModel
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from torchmeta.utils.gradient_based import gradient_update_parameters

from utils.average_meter import AverageMeter
from utils import utils as utils
from .common import WordEmbedding


class FUMI(nn.Module):
    def __init__(self,
                 n_way=5,
                 im_emb_dim=2048,
                 im_hid_dim=32,
                 text_encoder="BERT",
                 text_emb_dim=300,
                 text_hid_dim=1024,
                 dictionary=None,
                 pooling_strat="mean",
                 shared_feats=True):
        super(FUMI, self).__init__()
        self.n_way = n_way
        self.im_emb_dim = im_emb_dim
        self.im_hid_dim = im_hid_dim
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim  # only applicable if precomputed
        self.text_hid_dim = text_hid_dim
        self.dictionary = dictionary  # for word embeddings
        self.pooling_strat = pooling_strat

        if self.text_encoder_type == "BERT":
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_emb_dim = self.text_encoder.config.hidden_size
        elif self.text_encoder_type == "precomputed":
            self.text_encoder = nn.Identity()
        elif self.text_encoder_type == "w2v" or self.text_encoder_type == "glove":
            # load pretrained word embeddings as weights
            self.text_encoder = WordEmbedding(self.text_encoder_type,
                                              self.pooling_strat,
                                              self.dictionary)
            self.text_emb_dim = self.text_encoder.embedding_dim
        elif self.text_encoder_type == "rand":
            self.text_encoder = nn.Linear(self.text_emb_dim, self.text_emb_dim)
        else:
            raise NameError(f"{text_encoder} not allowed as text encoder")

        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.shared_feats = shared_feats
        if self.shared_feats:
            # Text embedding to image parameters
            self.net = nn.Sequential(
                nn.Linear(self.text_emb_dim, self.text_hid_dim),
                nn.ReLU(),
                nn.Linear(
                    self.text_hid_dim,
                    self.im_hid_dim  # Weights
                    + 1)  # Biases
            )
            # Bit of a hack to copy torch default weight initialisation
            self.first = nn.Linear(
                1,
                self.im_hid_dim * self.im_emb_dim  # Weights
                + self.im_hid_dim,  # Biases
                bias=False)
        else:
            # Text embedding to image parameters
            self.net = nn.Sequential(
                nn.Linear(self.text_emb_dim, self.text_hid_dim),
                nn.ReLU(),
                nn.Linear(
                    self.text_hid_dim,
                    self.im_hid_dim * (self.im_emb_dim + 1)  # Weights
                    + self.im_hid_dim + 1)  # Biases
            )

    def forward(self, text_embed, device):
        im_params = self.net(text_embed)
        if self.shared_feats:
            shared_params = self.first(torch.ones(1).to(device))
            bias_len = self.im_hid_dim + 1
            out = torch.empty(
                len(text_embed),
                self.im_hid_dim * (self.im_emb_dim + 1) + self.im_hid_dim +
                1).to(device)
            out[:, :bias_len - 1] = shared_params[:bias_len - 1]
            out[:, bias_len - 1] = im_params[:, 0]
            out[:, bias_len:-self.im_hid_dim] = shared_params[bias_len - 1:]
            out[:, -self.im_hid_dim:] = im_params[:, 1:]
            return out
        return im_params

    def evaluate(self, args, batch, optimizer, task="train"):
        """
        Evaluate batch on model

        Returns:
        - outer_loss: outer loop loss
        - acc: accuracy on query set
        """
        if task == "train":
            self.train()
            self.zero_grad()
        else:
            self.eval()

        # Support set
        train_inputs, train_targets = batch['train']
        train_inputs = [x.to(args.device) for x in train_inputs]
        # train_inputs = train_inputs[3].to(device=args.device)
        train_targets = train_targets.to(device=args.device)

        # Query set
        test_inputs, test_targets = batch['test']
        test_inputs = [x.to(args.device) for x in test_inputs]
        # test_inputs = test_inputs[3].to(device=args.device)
        test_targets = test_targets.to(device=args.device)
        test_preds = torch.zeros(test_targets.shape).to(device=args.device)

        # Unpack input
        if self.text_encoder_type == "BERT":
            _, train_texts, train_attn_masks, train_imss = train_inputs
            _, test_texts, test_attn_masks, test_imss = test_inputs
        else:
            _, train_texts, train_imss = train_inputs
            _, test_texts, test_imss = test_inputs

        outer_loss = torch.tensor(0., device=args.device)
        accuracy = torch.tensor(0., device=args.device)
        for task_idx, (train_target, test_target) in enumerate(
                zip(train_targets, test_targets)):
            n_steps = 0
            if task == "train":
                n_steps = args.num_train_adapt_steps
            else:
                n_steps = args.num_test_adapt_steps

            if self.text_encoder_type == "BERT":
                im_params = self.get_im_params(train_texts[task_idx],
                                               train_target, args.device,
                                               train_attn_masks[task_idx])
            else:
                im_params = self.get_im_params(train_texts[task_idx],
                                               train_target, args.device)

            for _ in range(n_steps):
                train_logit = self.im_forward(train_imss[task_idx], im_params)
                inner_loss = F.cross_entropy(train_logit, train_target)
                grads = torch.autograd.grad(inner_loss,
                                            im_params,
                                            create_graph=not args.first_order)
                im_params -= args.step_size * grads[0]

            test_logit = self.im_forward(test_imss[task_idx], im_params)
            _, test_preds[task_idx] = test_logit.max(dim=-1)

            outer_loss += F.cross_entropy(test_logit, test_target)

            with torch.no_grad():
                accuracy += get_accuracy(test_logit, test_target)

        outer_loss.div_(train_imss.shape[0])
        accuracy.div_(train_imss.shape[0])

        if task == "train":
            optimizer.zero_grad()
            outer_loss.backward()
            optimizer.step()

        return outer_loss.detach().cpu().numpy(), accuracy.detach().cpu(
        ).numpy(), test_preds, test_targets

    def get_im_params(self, text, targets, device, attn_mask=None):
        NK, seq_len = text.shape
        if self.text_encoder_type == "BERT":
            # Need to reshape batch for BERT input
            bert_output = self.text_encoder(text.view(-1, seq_len),
                                            attention_mask=attn_mask.view(
                                                -1, seq_len))
            # Get [CLS] token
            text_encoding = bert_output[1].view(NK, -1)  # (N*K x 768)
        elif self.text_encoder_type == "rand":
            # Get a random tensor as the encoding
            text_encoding = 2 * torch.rand(NK, self.text_emb_dim) - 1
        else:
            text_encoding = self.text_encoder(text.unsqueeze(0)).squeeze()

        # Transform to per-class descriptions
        class_text_enc = torch.empty(self.n_way, self.text_emb_dim).to(device)
        for i in range(self.n_way):
            class_text_enc[i] = text_encoding[(targets == i).nonzero(
                as_tuple=True)[0][0]]

        return self(class_text_enc, device)

    def im_forward(self, im_embeds, im_params):
        bias_len = self.im_hid_dim + 1
        b_im = torch.unsqueeze(im_params[:, :bias_len], 2)
        w_im = im_params[:, bias_len:].view(-1, self.im_emb_dim + 1,
                                            self.im_hid_dim)

        a = torch.matmul(im_embeds, w_im[:, :-1])
        h = F.relu(torch.transpose(a, 1, 2) + b_im[:, :-1])

        a_out = torch.matmul(torch.transpose(h, 1, 2),
                             torch.unsqueeze(w_im[:, -1], 2))
        out = torch.squeeze(a_out) + b_im[:, -1]
        return torch.transpose(out, 0, 1)


def training_run(args, model, optimizer, train_loader, val_loader,
                 max_test_batches):
    """
    FUMI training loop
    """

    best_loss, best_acc, _, _ = test_loop(args, model, val_loader, max_test_batches)
    print(f"\ninitial loss: {best_loss}, acc: {best_acc}")
    best_batch_idx = 0

    try:
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            train_loss, train_acc, _, _ = model.evaluate(args=args,
                                                   batch=batch,
                                                   optimizer=optimizer,
                                                   task="train")

            wandb.log(
                {
                    "train/acc": train_acc,
                    "train/loss": train_loss,
                    "num_episodes": (batch_idx + 1) * args.batch_size
                },
                step=batch_idx)

            # Eval on validation set periodically
            if batch_idx % args.eval_freq == 0 and batch_idx != 0:
                val_loss, val_acc, _, _ = test_loop(args, model, val_loader,
                                              max_test_batches)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                    best_batch_idx = batch_idx
                wandb.log({
                    "val/acc": val_acc,
                    "val/loss": val_loss
                },
                          step=batch_idx)

                checkpoint_dict = {
                    "batch_idx": batch_idx,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args)
                }
                utils.save_checkpoint(checkpoint_dict, is_best)

                print(
                    f"\nBatch {batch_idx+1}/{args.epochs}: \ntrain/loss: {train_loss}, train/acc: {train_acc}"
                    f"\nval/loss: {val_loss}, val/acc: {val_acc}")

            # break after max iters or early stopping
            if (batch_idx > args.epochs - 1) or (
                    args.patience > 0
                    and batch_idx - best_batch_idx > args.patience):
                break
    except KeyboardInterrupt:
        pass

    return model


def test_loop(args, model, test_loader, max_num_batches):
    """
    Evaluate model on val/test set.
    Returns:
    - avg_test_acc (float): average test accuracy per task
    - avg_test_loss (float): average test loss per task
    """

    avg_test_acc = AverageMeter()
    avg_test_loss = AverageMeter()
    test_preds = []
    test_targets = []
    for batch_idx, batch in enumerate(
            tqdm(test_loader, total=max_num_batches, position=0, leave=True)):
        test_loss, test_acc, preds, target = model.evaluate(args=args,
                                                            batch=batch,
                                                            optimizer=None,
                                                            task="test")
        avg_test_acc.update(test_acc)
        avg_test_loss.update(test_loss)
        test_preds.append(preds)
        test_targets.append(target)
        if batch_idx > max_num_batches - 1:
            break
    return avg_test_loss.avg, avg_test_acc.avg, test_preds, test_targets


def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())
