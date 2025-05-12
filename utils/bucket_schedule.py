import torch
import math, torch

def bucket_schedule(T, bucket_edges, alpha_bar_edges):
    u = torch.full((T+1,), float('nan'))
    for t_edge, a_edge in zip(bucket_edges, alpha_bar_edges):
        u[t_edge] = math.log(a_edge)

    # linear interpolation in log-ᾱ between edges
    for (ta, tb) in zip(bucket_edges[:-1], bucket_edges[1:]):
        ua, ub = u[ta], u[tb]
        L = tb - ta
        k = torch.arange(1, L+1)
        u[ta+1:tb+1] = ua + k * (ub - ua) / L

    alpha_bar = u.exp()
    alphas = alpha_bar[1:] / alpha_bar[:-1]
    betas  = 1.0 - alphas
    return betas

def get_ratio(Lm1, L):
    return (torch.polygamma(1, torch.tensor(Lm1)) - 1)/(torch.polygamma(1, torch.tensor(L)) - 1)

def make_betas():
    T = 1_000
    t_edges = [
        0,
        15,
        30,
        46,
        61,
        76,
        92,
        107,
        123,
        138,
        153,
        169,
        184,
        200,
        215,
        230,
        246,
        261,
        276,
        292,
        307,
        323,
        338,
        353,
        369,
        384,
        400,
        415,
        430,
        446,
        461,
        476,
        492,
        507,
        523,
        538,
        553,
        569,
        584,
        600,
        615,
        630,
        646,
        661,
        676,
        692,
        707,
        723,
        738,
        753,
        769,
        784,
        800,
        815,
        830,
        846,
        861,
        876,
        892,
        907,
        923,
        938,
        953,
        969,
        984,
        1_000,
    ]
    Ls = [*range(64, 0, -1)]
    Ls.insert(0, 100)
    Ls.append(0)
    alpha_bar_edges = [1.0]
    for i in range(len(Ls) - 1):
        L = Ls[i]
        Lm1 = Ls[i + 1]
        ratio = get_ratio(Lm1, L)
        alpha_bar_edges.append(alpha_bar_edges[-1] * ratio.item())

    alpha_bar_edges[-2] = 0.2
    alpha_bar_edges[-1] = 0.01

    alpha_bar_edges = torch.tensor(alpha_bar_edges)

    betas = bucket_schedule(T, t_edges, alpha_bar_edges)
    return betas

if __name__ == '__main__':
    make_betas()
    T = 1_000
    t_edges = [0,   15,   30,   46,   61,   76,   92,  107,  123,  138,  153,  169,
         184,  200,  215,  230,  246,  261,  276,  292,  307,  323,  338,  353,
         369,  384,  400,  415,  430,  446,  461,  476,  492,  507,  523,  538,
         553,  569,  584,  600,  615,  630,  646,  661,  676,  692,  707,  723,
         738,  753,  769,  784,  800,  815,  830,  846,  861,  876,  892,  907,
         923,  938,  953,  969,  984, 1_000]
    Ls = [*range(64, 0, -1)]
    Ls.insert(0, 100)
    Ls.append(0)
    alpha_bar_edges = [1.0]
    for i in range(len(Ls)-1):
        L = Ls[i]
        Lm1 = Ls[i+1]
        ratio = get_ratio(Lm1, L)
        alpha_bar_edges.append(alpha_bar_edges[-1] * ratio.item())
        
    alpha_bar_edges[-2] = 0.2
    alpha_bar_edges[-1] = 0.01
    
    alpha_bar_edges = torch.tensor(alpha_bar_edges)
        
    
    betas  = bucket_schedule(T, t_edges, alpha_bar_edges)
    
    assert (0 < betas).all() and (betas < 0.02).all()
    assert abs((1-betas).cumprod(0)[t_edges] - torch.tensor(alpha_bar_edges)).max() < 1e-6
