

import torch

from util import batch_n_noise as attack_initialization


def basic_iterative_method(model: torch.nn.Module, image : torch.Tensor, eps : float=0.001, iter : int=5, batch_size : int=10) -> None:
    """Performs the Basic Iterative Method over FGSM

    Args:
        model (torch.nn.Module): Face Embedder network with added Normalization layer
        image (torch.Tensor): Torch tensor of input image
        eps (float, optional): Parameter controlling the amount of noise added by BIM. Defaults to 0.001.
        iter (int, optional): Number of iterations of BIM to perform. Defaults to 5.
        batch_size (int, optional): Number of directions to perform BIM in. Defaults to 10.

    Returns:
        list: List of similarities obtained from BIM iterations.
    """
    
    model = model.cuda().eval()
    image = image.cuda()

    noisy_batch = attack_initialization(image, eps, batch_size)

    base_emb = model(image.unsqueeze(0))

    cos_loss = torch.nn.CosineEmbeddingLoss()
    cos_sim = torch.nn.CosineSimilarity(dim=1)

    similarities = []

    for _ in range(iter):

        image.requires_grad = noisy_batch.requires_grad = True

        noisy_embs = model(noisy_batch)

        loss = cos_loss(input1=base_emb, input2=noisy_embs, target=torch.Tensor([1]).cuda())

        model.zero_grad()
        loss.backward()

        grad = noisy_batch.grad.data

        noisy_batch = torch.clamp(noisy_batch + eps * grad.sign(), -1., 1.)
        noisy_batch, image, noisy_embs, base_emb = noisy_batch.detach(), image.detach(), noisy_embs.detach(), base_emb.detach()

        similarities.append(cos_sim(base_emb.detach(), noisy_embs.detach()).cpu().numpy())

    return similarities