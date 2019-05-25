import torch


def secured_softmax(input, dim, mask, eps=1e-10):
    assert mask.size() == input.size(), f'{mask.size()}!={input.size()}'
    input_max = input.max(dim, keepdim=True)[0]

    # input_max=input_max(dim)

    masked = mask.min(dim)[0].unsqueeze(dim).float()  # 1 means all masked
    eps = masked * eps

    ex = torch.exp(input - input_max)
    ex.data.masked_fill_(mask, 0)
    denorm = ex.sum(dim, keepdim=True) + eps
    return ex / denorm
