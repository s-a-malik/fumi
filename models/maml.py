import torch
import torch.nn.functional as F

from tqdm import tqdm
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from torchmeta.utils.gradient_based import gradient_update_parameters


def PureImageNetwork(MetaModule):
    def __init__(self, im_embed_dim=512, n_way=5, hidden=64):
        super(PureImageNetwork, self).__init__()
        self.im_embed_dim = im_embed_dim
        self.n_way = n_way
        self.hidden = hidden

        self.net = MetaSequential(
            MetaLinear(im_embed_dim, hidden),
            MetaLinear(hidden, n_way)
        )

    def forward(self, inputs):
      logits = self.net(inputs)
      return logits

def train_maml(model, dataloader, step_size, first_order, batch_size):
    """
    Args:
    model -- MetaModule
    dataloader -- Torchmeta BatchMetaDataLoader
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    meta_opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tqdm(dataloader) as pbar:
        for batch in pbar:
            model.zero_grad()

            train_inputs, train_targets = batch['train']
            train_inputs = train_inputs.to(device=device)
            train_targets = train_targets.to(device=device)

            test_inputs, test_targets = batch['test']
            test_inputs = test_inputs.to(device=device)
            test_targets = test_targets.to(device=device)

            outer_loss = torch.tensor(0., device=device)
            accuracy = torch.tensor(0., device=device)
            for task_idx, (train_input, train_target, test_input,
                           test_target) in enumerate(zip(train_inputs, train_targets,
                                                         test_inputs, test_targets)):
                train_logit = model(train_input)
                inner_loss = F.cross_entropy(train_logit, train_target)

                model.zero_grad()
                params = gradient_update_parameters(model,
                                                    inner_loss,
                                                    step_size=step_size,
                                                    first_order=first_order)

                test_logit = model(test_input, params=params)
                outer_loss += F.cross_entropy(test_logit, test_target)

                with torch.no_grad():
                    accuracy += get_accuracy(test_logit, test_target)

            outer_loss.div_(batch_size)
            accuracy.div_(batch_size)

            outer_loss.backward()
            meta_opt.step()

            pbar.set_postfix(accuracy='{0:.4f}'.format(accuracy.item()))

def get_accuracy(logits, targets):
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())
