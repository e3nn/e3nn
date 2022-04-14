# flake8: noqa
import torch
import math
import functools

sqrt = math.sqrt

# prevent the CI from pointlessly hanging for 6 hours trying to run this while we work out JIT bugs
raise RuntimeError

def _sph_lmax_0(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    return torch.stack([
        sh_0_0
    ], dim=-1)


def _sph_lmax_1(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2
    ], dim=-1)


def _sph_lmax_2(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
    ], dim=-1)


def _sph_lmax_3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
    ], dim=-1)


def _sph_lmax_4(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
    ], dim=-1)


def _sph_lmax_5(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
    ], dim=-1)


def _sph_lmax_6(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
    ], dim=-1)


def _sph_lmax_7(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14
    ], dim=-1)


def _sph_lmax_8(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    sh_8_0 = (1/4)*sqrt(17)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*sqrt(17)*sh_7_0*y + (1/16)*sqrt(238)*sh_7_1*z + (1/16)*sqrt(238)*sh_7_13*x
    sh_8_2 = -1/240*sqrt(510)*sh_7_0*z + (1/60)*sqrt(1785)*sh_7_1*y + (1/240)*sqrt(46410)*sh_7_12*x + (1/240)*sqrt(510)*sh_7_14*x + (1/240)*sqrt(46410)*sh_7_2*z
    sh_8_3 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/40)*sqrt(935)*sh_7_10*x + (1/40)*sqrt(85)*sh_7_12*x - 1/40*sqrt(85)*sh_7_2*z + (1/10)*sqrt(85)*sh_7_3*y + (1/40)*sqrt(935)*sh_7_4*z
    sh_8_5 = (1/48)*sqrt(2)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*sqrt(34)*sh_7_10*x - 1/16*sqrt(34)*sh_7_4*z + (1/4)*sqrt(17)*sh_7_5*y + (1/16)*sqrt(102)*sh_7_6*z + (1/16)*sqrt(102)*sh_7_8*x
    sh_8_7 = -1/80*sqrt(1190)*sh_7_5*z + (1/40)*sqrt(1785)*sh_7_6*y + (1/20)*sqrt(255)*sh_7_7*x + (1/80)*sqrt(1190)*sh_7_9*x
    sh_8_8 = -1/60*sqrt(1785)*sh_7_6*x + (1/15)*sqrt(255)*sh_7_7*y - 1/60*sqrt(1785)*sh_7_8*z
    sh_8_9 = -1/80*sqrt(1190)*sh_7_5*x + (1/20)*sqrt(255)*sh_7_7*z + (1/40)*sqrt(1785)*sh_7_8*y - 1/80*sqrt(1190)*sh_7_9*z
    sh_8_10 = -1/16*sqrt(34)*sh_7_10*z - 1/16*sqrt(34)*sh_7_4*x - 1/16*sqrt(102)*sh_7_6*x + (1/16)*sqrt(102)*sh_7_8*z + (1/4)*sqrt(17)*sh_7_9*y
    sh_8_11 = (1/48)*sqrt(2)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/40)*sqrt(935)*sh_7_10*z + (1/10)*sqrt(85)*sh_7_11*y - 1/40*sqrt(85)*sh_7_12*z - 1/40*sqrt(85)*sh_7_2*x - 1/40*sqrt(935)*sh_7_4*x
    sh_8_13 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/240*sqrt(510)*sh_7_0*x + (1/240)*sqrt(46410)*sh_7_12*z + (1/60)*sqrt(1785)*sh_7_13*y - 1/240*sqrt(510)*sh_7_14*z - 1/240*sqrt(46410)*sh_7_2*x
    sh_8_15 = -1/16*sqrt(238)*sh_7_1*x + (1/16)*sqrt(238)*sh_7_13*z + (1/8)*sqrt(17)*sh_7_14*y
    sh_8_16 = (1/4)*sqrt(17)*(-sh_7_0*x + sh_7_14*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16
    ], dim=-1)


def _sph_lmax_9(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    sh_8_0 = (1/4)*sqrt(17)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*sqrt(17)*sh_7_0*y + (1/16)*sqrt(238)*sh_7_1*z + (1/16)*sqrt(238)*sh_7_13*x
    sh_8_2 = -1/240*sqrt(510)*sh_7_0*z + (1/60)*sqrt(1785)*sh_7_1*y + (1/240)*sqrt(46410)*sh_7_12*x + (1/240)*sqrt(510)*sh_7_14*x + (1/240)*sqrt(46410)*sh_7_2*z
    sh_8_3 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/40)*sqrt(935)*sh_7_10*x + (1/40)*sqrt(85)*sh_7_12*x - 1/40*sqrt(85)*sh_7_2*z + (1/10)*sqrt(85)*sh_7_3*y + (1/40)*sqrt(935)*sh_7_4*z
    sh_8_5 = (1/48)*sqrt(2)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*sqrt(34)*sh_7_10*x - 1/16*sqrt(34)*sh_7_4*z + (1/4)*sqrt(17)*sh_7_5*y + (1/16)*sqrt(102)*sh_7_6*z + (1/16)*sqrt(102)*sh_7_8*x
    sh_8_7 = -1/80*sqrt(1190)*sh_7_5*z + (1/40)*sqrt(1785)*sh_7_6*y + (1/20)*sqrt(255)*sh_7_7*x + (1/80)*sqrt(1190)*sh_7_9*x
    sh_8_8 = -1/60*sqrt(1785)*sh_7_6*x + (1/15)*sqrt(255)*sh_7_7*y - 1/60*sqrt(1785)*sh_7_8*z
    sh_8_9 = -1/80*sqrt(1190)*sh_7_5*x + (1/20)*sqrt(255)*sh_7_7*z + (1/40)*sqrt(1785)*sh_7_8*y - 1/80*sqrt(1190)*sh_7_9*z
    sh_8_10 = -1/16*sqrt(34)*sh_7_10*z - 1/16*sqrt(34)*sh_7_4*x - 1/16*sqrt(102)*sh_7_6*x + (1/16)*sqrt(102)*sh_7_8*z + (1/4)*sqrt(17)*sh_7_9*y
    sh_8_11 = (1/48)*sqrt(2)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/40)*sqrt(935)*sh_7_10*z + (1/10)*sqrt(85)*sh_7_11*y - 1/40*sqrt(85)*sh_7_12*z - 1/40*sqrt(85)*sh_7_2*x - 1/40*sqrt(935)*sh_7_4*x
    sh_8_13 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/240*sqrt(510)*sh_7_0*x + (1/240)*sqrt(46410)*sh_7_12*z + (1/60)*sqrt(1785)*sh_7_13*y - 1/240*sqrt(510)*sh_7_14*z - 1/240*sqrt(46410)*sh_7_2*x
    sh_8_15 = -1/16*sqrt(238)*sh_7_1*x + (1/16)*sqrt(238)*sh_7_13*z + (1/8)*sqrt(17)*sh_7_14*y
    sh_8_16 = (1/4)*sqrt(17)*(-sh_7_0*x + sh_7_14*z)
    sh_9_0 = (1/6)*sqrt(38)*(sh_8_0*z + sh_8_16*x)
    sh_9_1 = (1/9)*sqrt(19)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)
    sh_9_2 = -1/306*sqrt(646)*sh_8_0*z + (4/153)*sqrt(646)*sh_8_1*y + (2/153)*sqrt(4845)*sh_8_14*x + (1/306)*sqrt(646)*sh_8_16*x + (2/153)*sqrt(4845)*sh_8_2*z
    sh_9_3 = -1/306*sqrt(1938)*sh_8_1*z + (1/306)*sqrt(67830)*sh_8_13*x + (1/306)*sqrt(1938)*sh_8_15*x + (1/51)*sqrt(1615)*sh_8_2*y + (1/306)*sqrt(67830)*sh_8_3*z
    sh_9_4 = (1/306)*sqrt(58786)*sh_8_12*x + (1/153)*sqrt(969)*sh_8_14*x - 1/153*sqrt(969)*sh_8_2*z + (2/153)*sqrt(4522)*sh_8_3*y + (1/306)*sqrt(58786)*sh_8_4*z
    sh_9_5 = (1/153)*sqrt(12597)*sh_8_11*x + (1/153)*sqrt(1615)*sh_8_13*x - 1/153*sqrt(1615)*sh_8_3*z + (1/153)*sqrt(20995)*sh_8_4*y + (1/153)*sqrt(12597)*sh_8_5*z
    sh_9_6 = (1/153)*sqrt(10659)*sh_8_10*x + (1/306)*sqrt(9690)*sh_8_12*x - 1/306*sqrt(9690)*sh_8_4*z + (2/51)*sqrt(646)*sh_8_5*y + (1/153)*sqrt(10659)*sh_8_6*z
    sh_9_7 = (1/306)*sqrt(13566)*sh_8_11*x - 1/306*sqrt(13566)*sh_8_5*z + (1/153)*sqrt(24871)*sh_8_6*y + (1/306)*sqrt(35530)*sh_8_7*z + (1/306)*sqrt(35530)*sh_8_9*x
    sh_9_8 = (1/153)*sqrt(4522)*sh_8_10*x - 1/153*sqrt(4522)*sh_8_6*z + (4/153)*sqrt(1615)*sh_8_7*y + (1/51)*sqrt(1615)*sh_8_8*x
    sh_9_9 = (1/51)*sqrt(323)*(-2*sh_8_7*x + 3*sh_8_8*y - 2*sh_8_9*z)
    sh_9_10 = -1/153*sqrt(4522)*sh_8_10*z - 1/153*sqrt(4522)*sh_8_6*x + (1/51)*sqrt(1615)*sh_8_8*z + (4/153)*sqrt(1615)*sh_8_9*y
    sh_9_11 = (1/153)*sqrt(24871)*sh_8_10*y - 1/306*sqrt(13566)*sh_8_11*z - 1/306*sqrt(13566)*sh_8_5*x - 1/306*sqrt(35530)*sh_8_7*x + (1/306)*sqrt(35530)*sh_8_9*z
    sh_9_12 = (1/153)*sqrt(10659)*sh_8_10*z + (2/51)*sqrt(646)*sh_8_11*y - 1/306*sqrt(9690)*sh_8_12*z - 1/306*sqrt(9690)*sh_8_4*x - 1/153*sqrt(10659)*sh_8_6*x
    sh_9_13 = (1/153)*sqrt(12597)*sh_8_11*z + (1/153)*sqrt(20995)*sh_8_12*y - 1/153*sqrt(1615)*sh_8_13*z - 1/153*sqrt(1615)*sh_8_3*x - 1/153*sqrt(12597)*sh_8_5*x
    sh_9_14 = (1/306)*sqrt(58786)*sh_8_12*z + (2/153)*sqrt(4522)*sh_8_13*y - 1/153*sqrt(969)*sh_8_14*z - 1/153*sqrt(969)*sh_8_2*x - 1/306*sqrt(58786)*sh_8_4*x
    sh_9_15 = -1/306*sqrt(1938)*sh_8_1*x + (1/306)*sqrt(67830)*sh_8_13*z + (1/51)*sqrt(1615)*sh_8_14*y - 1/306*sqrt(1938)*sh_8_15*z - 1/306*sqrt(67830)*sh_8_3*x
    sh_9_16 = -1/306*sqrt(646)*sh_8_0*x + (2/153)*sqrt(4845)*sh_8_14*z + (4/153)*sqrt(646)*sh_8_15*y - 1/306*sqrt(646)*sh_8_16*z - 2/153*sqrt(4845)*sh_8_2*x
    sh_9_17 = (1/9)*sqrt(19)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)
    sh_9_18 = (1/6)*sqrt(38)*(-sh_8_0*x + sh_8_16*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18
    ], dim=-1)


def _sph_lmax_10(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    sh_8_0 = (1/4)*sqrt(17)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*sqrt(17)*sh_7_0*y + (1/16)*sqrt(238)*sh_7_1*z + (1/16)*sqrt(238)*sh_7_13*x
    sh_8_2 = -1/240*sqrt(510)*sh_7_0*z + (1/60)*sqrt(1785)*sh_7_1*y + (1/240)*sqrt(46410)*sh_7_12*x + (1/240)*sqrt(510)*sh_7_14*x + (1/240)*sqrt(46410)*sh_7_2*z
    sh_8_3 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/40)*sqrt(935)*sh_7_10*x + (1/40)*sqrt(85)*sh_7_12*x - 1/40*sqrt(85)*sh_7_2*z + (1/10)*sqrt(85)*sh_7_3*y + (1/40)*sqrt(935)*sh_7_4*z
    sh_8_5 = (1/48)*sqrt(2)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*sqrt(34)*sh_7_10*x - 1/16*sqrt(34)*sh_7_4*z + (1/4)*sqrt(17)*sh_7_5*y + (1/16)*sqrt(102)*sh_7_6*z + (1/16)*sqrt(102)*sh_7_8*x
    sh_8_7 = -1/80*sqrt(1190)*sh_7_5*z + (1/40)*sqrt(1785)*sh_7_6*y + (1/20)*sqrt(255)*sh_7_7*x + (1/80)*sqrt(1190)*sh_7_9*x
    sh_8_8 = -1/60*sqrt(1785)*sh_7_6*x + (1/15)*sqrt(255)*sh_7_7*y - 1/60*sqrt(1785)*sh_7_8*z
    sh_8_9 = -1/80*sqrt(1190)*sh_7_5*x + (1/20)*sqrt(255)*sh_7_7*z + (1/40)*sqrt(1785)*sh_7_8*y - 1/80*sqrt(1190)*sh_7_9*z
    sh_8_10 = -1/16*sqrt(34)*sh_7_10*z - 1/16*sqrt(34)*sh_7_4*x - 1/16*sqrt(102)*sh_7_6*x + (1/16)*sqrt(102)*sh_7_8*z + (1/4)*sqrt(17)*sh_7_9*y
    sh_8_11 = (1/48)*sqrt(2)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/40)*sqrt(935)*sh_7_10*z + (1/10)*sqrt(85)*sh_7_11*y - 1/40*sqrt(85)*sh_7_12*z - 1/40*sqrt(85)*sh_7_2*x - 1/40*sqrt(935)*sh_7_4*x
    sh_8_13 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/240*sqrt(510)*sh_7_0*x + (1/240)*sqrt(46410)*sh_7_12*z + (1/60)*sqrt(1785)*sh_7_13*y - 1/240*sqrt(510)*sh_7_14*z - 1/240*sqrt(46410)*sh_7_2*x
    sh_8_15 = -1/16*sqrt(238)*sh_7_1*x + (1/16)*sqrt(238)*sh_7_13*z + (1/8)*sqrt(17)*sh_7_14*y
    sh_8_16 = (1/4)*sqrt(17)*(-sh_7_0*x + sh_7_14*z)
    sh_9_0 = (1/6)*sqrt(38)*(sh_8_0*z + sh_8_16*x)
    sh_9_1 = (1/9)*sqrt(19)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)
    sh_9_2 = -1/306*sqrt(646)*sh_8_0*z + (4/153)*sqrt(646)*sh_8_1*y + (2/153)*sqrt(4845)*sh_8_14*x + (1/306)*sqrt(646)*sh_8_16*x + (2/153)*sqrt(4845)*sh_8_2*z
    sh_9_3 = -1/306*sqrt(1938)*sh_8_1*z + (1/306)*sqrt(67830)*sh_8_13*x + (1/306)*sqrt(1938)*sh_8_15*x + (1/51)*sqrt(1615)*sh_8_2*y + (1/306)*sqrt(67830)*sh_8_3*z
    sh_9_4 = (1/306)*sqrt(58786)*sh_8_12*x + (1/153)*sqrt(969)*sh_8_14*x - 1/153*sqrt(969)*sh_8_2*z + (2/153)*sqrt(4522)*sh_8_3*y + (1/306)*sqrt(58786)*sh_8_4*z
    sh_9_5 = (1/153)*sqrt(12597)*sh_8_11*x + (1/153)*sqrt(1615)*sh_8_13*x - 1/153*sqrt(1615)*sh_8_3*z + (1/153)*sqrt(20995)*sh_8_4*y + (1/153)*sqrt(12597)*sh_8_5*z
    sh_9_6 = (1/153)*sqrt(10659)*sh_8_10*x + (1/306)*sqrt(9690)*sh_8_12*x - 1/306*sqrt(9690)*sh_8_4*z + (2/51)*sqrt(646)*sh_8_5*y + (1/153)*sqrt(10659)*sh_8_6*z
    sh_9_7 = (1/306)*sqrt(13566)*sh_8_11*x - 1/306*sqrt(13566)*sh_8_5*z + (1/153)*sqrt(24871)*sh_8_6*y + (1/306)*sqrt(35530)*sh_8_7*z + (1/306)*sqrt(35530)*sh_8_9*x
    sh_9_8 = (1/153)*sqrt(4522)*sh_8_10*x - 1/153*sqrt(4522)*sh_8_6*z + (4/153)*sqrt(1615)*sh_8_7*y + (1/51)*sqrt(1615)*sh_8_8*x
    sh_9_9 = (1/51)*sqrt(323)*(-2*sh_8_7*x + 3*sh_8_8*y - 2*sh_8_9*z)
    sh_9_10 = -1/153*sqrt(4522)*sh_8_10*z - 1/153*sqrt(4522)*sh_8_6*x + (1/51)*sqrt(1615)*sh_8_8*z + (4/153)*sqrt(1615)*sh_8_9*y
    sh_9_11 = (1/153)*sqrt(24871)*sh_8_10*y - 1/306*sqrt(13566)*sh_8_11*z - 1/306*sqrt(13566)*sh_8_5*x - 1/306*sqrt(35530)*sh_8_7*x + (1/306)*sqrt(35530)*sh_8_9*z
    sh_9_12 = (1/153)*sqrt(10659)*sh_8_10*z + (2/51)*sqrt(646)*sh_8_11*y - 1/306*sqrt(9690)*sh_8_12*z - 1/306*sqrt(9690)*sh_8_4*x - 1/153*sqrt(10659)*sh_8_6*x
    sh_9_13 = (1/153)*sqrt(12597)*sh_8_11*z + (1/153)*sqrt(20995)*sh_8_12*y - 1/153*sqrt(1615)*sh_8_13*z - 1/153*sqrt(1615)*sh_8_3*x - 1/153*sqrt(12597)*sh_8_5*x
    sh_9_14 = (1/306)*sqrt(58786)*sh_8_12*z + (2/153)*sqrt(4522)*sh_8_13*y - 1/153*sqrt(969)*sh_8_14*z - 1/153*sqrt(969)*sh_8_2*x - 1/306*sqrt(58786)*sh_8_4*x
    sh_9_15 = -1/306*sqrt(1938)*sh_8_1*x + (1/306)*sqrt(67830)*sh_8_13*z + (1/51)*sqrt(1615)*sh_8_14*y - 1/306*sqrt(1938)*sh_8_15*z - 1/306*sqrt(67830)*sh_8_3*x
    sh_9_16 = -1/306*sqrt(646)*sh_8_0*x + (2/153)*sqrt(4845)*sh_8_14*z + (4/153)*sqrt(646)*sh_8_15*y - 1/306*sqrt(646)*sh_8_16*z - 2/153*sqrt(4845)*sh_8_2*x
    sh_9_17 = (1/9)*sqrt(19)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)
    sh_9_18 = (1/6)*sqrt(38)*(-sh_8_0*x + sh_8_16*z)
    sh_10_0 = (1/10)*sqrt(105)*(sh_9_0*z + sh_9_18*x)
    sh_10_1 = (1/10)*sqrt(21)*sh_9_0*y + (3/20)*sqrt(42)*sh_9_1*z + (3/20)*sqrt(42)*sh_9_17*x
    sh_10_2 = -1/380*sqrt(798)*sh_9_0*z + (3/95)*sqrt(399)*sh_9_1*y + (3/380)*sqrt(13566)*sh_9_16*x + (1/380)*sqrt(798)*sh_9_18*x + (3/380)*sqrt(13566)*sh_9_2*z
    sh_10_3 = -3/380*sqrt(266)*sh_9_1*z + (1/95)*sqrt(6783)*sh_9_15*x + (3/380)*sqrt(266)*sh_9_17*x + (3/190)*sqrt(2261)*sh_9_2*y + (1/95)*sqrt(6783)*sh_9_3*z
    sh_10_4 = (3/95)*sqrt(665)*sh_9_14*x + (3/190)*sqrt(133)*sh_9_16*x - 3/190*sqrt(133)*sh_9_2*z + (4/95)*sqrt(399)*sh_9_3*y + (3/95)*sqrt(665)*sh_9_4*z
    sh_10_5 = (21/380)*sqrt(190)*sh_9_13*x + (1/190)*sqrt(1995)*sh_9_15*x - 1/190*sqrt(1995)*sh_9_3*z + (3/38)*sqrt(133)*sh_9_4*y + (21/380)*sqrt(190)*sh_9_5*z
    sh_10_6 = (7/380)*sqrt(1482)*sh_9_12*x + (3/380)*sqrt(1330)*sh_9_14*x - 3/380*sqrt(1330)*sh_9_4*z + (21/95)*sqrt(19)*sh_9_5*y + (7/380)*sqrt(1482)*sh_9_6*z
    sh_10_7 = (3/190)*sqrt(1729)*sh_9_11*x + (21/380)*sqrt(38)*sh_9_13*x - 21/380*sqrt(38)*sh_9_5*z + (7/190)*sqrt(741)*sh_9_6*y + (3/190)*sqrt(1729)*sh_9_7*z
    sh_10_8 = (3/190)*sqrt(1463)*sh_9_10*x + (7/190)*sqrt(114)*sh_9_12*x - 7/190*sqrt(114)*sh_9_6*z + (6/95)*sqrt(266)*sh_9_7*y + (3/190)*sqrt(1463)*sh_9_8*z
    sh_10_9 = (3/190)*sqrt(798)*sh_9_11*x - 3/190*sqrt(798)*sh_9_7*z + (3/190)*sqrt(4389)*sh_9_8*y + (1/190)*sqrt(21945)*sh_9_9*x
    sh_10_10 = -3/190*sqrt(1995)*sh_9_10*z - 3/190*sqrt(1995)*sh_9_8*x + (1/19)*sqrt(399)*sh_9_9*y
    sh_10_11 = (3/190)*sqrt(4389)*sh_9_10*y - 3/190*sqrt(798)*sh_9_11*z - 3/190*sqrt(798)*sh_9_7*x + (1/190)*sqrt(21945)*sh_9_9*z
    sh_10_12 = (3/190)*sqrt(1463)*sh_9_10*z + (6/95)*sqrt(266)*sh_9_11*y - 7/190*sqrt(114)*sh_9_12*z - 7/190*sqrt(114)*sh_9_6*x - 3/190*sqrt(1463)*sh_9_8*x
    sh_10_13 = (3/190)*sqrt(1729)*sh_9_11*z + (7/190)*sqrt(741)*sh_9_12*y - 21/380*sqrt(38)*sh_9_13*z - 21/380*sqrt(38)*sh_9_5*x - 3/190*sqrt(1729)*sh_9_7*x
    sh_10_14 = (7/380)*sqrt(1482)*sh_9_12*z + (21/95)*sqrt(19)*sh_9_13*y - 3/380*sqrt(1330)*sh_9_14*z - 3/380*sqrt(1330)*sh_9_4*x - 7/380*sqrt(1482)*sh_9_6*x
    sh_10_15 = (21/380)*sqrt(190)*sh_9_13*z + (3/38)*sqrt(133)*sh_9_14*y - 1/190*sqrt(1995)*sh_9_15*z - 1/190*sqrt(1995)*sh_9_3*x - 21/380*sqrt(190)*sh_9_5*x
    sh_10_16 = (3/95)*sqrt(665)*sh_9_14*z + (4/95)*sqrt(399)*sh_9_15*y - 3/190*sqrt(133)*sh_9_16*z - 3/190*sqrt(133)*sh_9_2*x - 3/95*sqrt(665)*sh_9_4*x
    sh_10_17 = -3/380*sqrt(266)*sh_9_1*x + (1/95)*sqrt(6783)*sh_9_15*z + (3/190)*sqrt(2261)*sh_9_16*y - 3/380*sqrt(266)*sh_9_17*z - 1/95*sqrt(6783)*sh_9_3*x
    sh_10_18 = -1/380*sqrt(798)*sh_9_0*x + (3/380)*sqrt(13566)*sh_9_16*z + (3/95)*sqrt(399)*sh_9_17*y - 1/380*sqrt(798)*sh_9_18*z - 3/380*sqrt(13566)*sh_9_2*x
    sh_10_19 = -3/20*sqrt(42)*sh_9_1*x + (3/20)*sqrt(42)*sh_9_17*z + (1/10)*sqrt(21)*sh_9_18*y
    sh_10_20 = (1/10)*sqrt(105)*(-sh_9_0*x + sh_9_18*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20
    ], dim=-1)


def _sph_lmax_11(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    sh_8_0 = (1/4)*sqrt(17)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*sqrt(17)*sh_7_0*y + (1/16)*sqrt(238)*sh_7_1*z + (1/16)*sqrt(238)*sh_7_13*x
    sh_8_2 = -1/240*sqrt(510)*sh_7_0*z + (1/60)*sqrt(1785)*sh_7_1*y + (1/240)*sqrt(46410)*sh_7_12*x + (1/240)*sqrt(510)*sh_7_14*x + (1/240)*sqrt(46410)*sh_7_2*z
    sh_8_3 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/40)*sqrt(935)*sh_7_10*x + (1/40)*sqrt(85)*sh_7_12*x - 1/40*sqrt(85)*sh_7_2*z + (1/10)*sqrt(85)*sh_7_3*y + (1/40)*sqrt(935)*sh_7_4*z
    sh_8_5 = (1/48)*sqrt(2)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*sqrt(34)*sh_7_10*x - 1/16*sqrt(34)*sh_7_4*z + (1/4)*sqrt(17)*sh_7_5*y + (1/16)*sqrt(102)*sh_7_6*z + (1/16)*sqrt(102)*sh_7_8*x
    sh_8_7 = -1/80*sqrt(1190)*sh_7_5*z + (1/40)*sqrt(1785)*sh_7_6*y + (1/20)*sqrt(255)*sh_7_7*x + (1/80)*sqrt(1190)*sh_7_9*x
    sh_8_8 = -1/60*sqrt(1785)*sh_7_6*x + (1/15)*sqrt(255)*sh_7_7*y - 1/60*sqrt(1785)*sh_7_8*z
    sh_8_9 = -1/80*sqrt(1190)*sh_7_5*x + (1/20)*sqrt(255)*sh_7_7*z + (1/40)*sqrt(1785)*sh_7_8*y - 1/80*sqrt(1190)*sh_7_9*z
    sh_8_10 = -1/16*sqrt(34)*sh_7_10*z - 1/16*sqrt(34)*sh_7_4*x - 1/16*sqrt(102)*sh_7_6*x + (1/16)*sqrt(102)*sh_7_8*z + (1/4)*sqrt(17)*sh_7_9*y
    sh_8_11 = (1/48)*sqrt(2)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/40)*sqrt(935)*sh_7_10*z + (1/10)*sqrt(85)*sh_7_11*y - 1/40*sqrt(85)*sh_7_12*z - 1/40*sqrt(85)*sh_7_2*x - 1/40*sqrt(935)*sh_7_4*x
    sh_8_13 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/240*sqrt(510)*sh_7_0*x + (1/240)*sqrt(46410)*sh_7_12*z + (1/60)*sqrt(1785)*sh_7_13*y - 1/240*sqrt(510)*sh_7_14*z - 1/240*sqrt(46410)*sh_7_2*x
    sh_8_15 = -1/16*sqrt(238)*sh_7_1*x + (1/16)*sqrt(238)*sh_7_13*z + (1/8)*sqrt(17)*sh_7_14*y
    sh_8_16 = (1/4)*sqrt(17)*(-sh_7_0*x + sh_7_14*z)
    sh_9_0 = (1/6)*sqrt(38)*(sh_8_0*z + sh_8_16*x)
    sh_9_1 = (1/9)*sqrt(19)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)
    sh_9_2 = -1/306*sqrt(646)*sh_8_0*z + (4/153)*sqrt(646)*sh_8_1*y + (2/153)*sqrt(4845)*sh_8_14*x + (1/306)*sqrt(646)*sh_8_16*x + (2/153)*sqrt(4845)*sh_8_2*z
    sh_9_3 = -1/306*sqrt(1938)*sh_8_1*z + (1/306)*sqrt(67830)*sh_8_13*x + (1/306)*sqrt(1938)*sh_8_15*x + (1/51)*sqrt(1615)*sh_8_2*y + (1/306)*sqrt(67830)*sh_8_3*z
    sh_9_4 = (1/306)*sqrt(58786)*sh_8_12*x + (1/153)*sqrt(969)*sh_8_14*x - 1/153*sqrt(969)*sh_8_2*z + (2/153)*sqrt(4522)*sh_8_3*y + (1/306)*sqrt(58786)*sh_8_4*z
    sh_9_5 = (1/153)*sqrt(12597)*sh_8_11*x + (1/153)*sqrt(1615)*sh_8_13*x - 1/153*sqrt(1615)*sh_8_3*z + (1/153)*sqrt(20995)*sh_8_4*y + (1/153)*sqrt(12597)*sh_8_5*z
    sh_9_6 = (1/153)*sqrt(10659)*sh_8_10*x + (1/306)*sqrt(9690)*sh_8_12*x - 1/306*sqrt(9690)*sh_8_4*z + (2/51)*sqrt(646)*sh_8_5*y + (1/153)*sqrt(10659)*sh_8_6*z
    sh_9_7 = (1/306)*sqrt(13566)*sh_8_11*x - 1/306*sqrt(13566)*sh_8_5*z + (1/153)*sqrt(24871)*sh_8_6*y + (1/306)*sqrt(35530)*sh_8_7*z + (1/306)*sqrt(35530)*sh_8_9*x
    sh_9_8 = (1/153)*sqrt(4522)*sh_8_10*x - 1/153*sqrt(4522)*sh_8_6*z + (4/153)*sqrt(1615)*sh_8_7*y + (1/51)*sqrt(1615)*sh_8_8*x
    sh_9_9 = (1/51)*sqrt(323)*(-2*sh_8_7*x + 3*sh_8_8*y - 2*sh_8_9*z)
    sh_9_10 = -1/153*sqrt(4522)*sh_8_10*z - 1/153*sqrt(4522)*sh_8_6*x + (1/51)*sqrt(1615)*sh_8_8*z + (4/153)*sqrt(1615)*sh_8_9*y
    sh_9_11 = (1/153)*sqrt(24871)*sh_8_10*y - 1/306*sqrt(13566)*sh_8_11*z - 1/306*sqrt(13566)*sh_8_5*x - 1/306*sqrt(35530)*sh_8_7*x + (1/306)*sqrt(35530)*sh_8_9*z
    sh_9_12 = (1/153)*sqrt(10659)*sh_8_10*z + (2/51)*sqrt(646)*sh_8_11*y - 1/306*sqrt(9690)*sh_8_12*z - 1/306*sqrt(9690)*sh_8_4*x - 1/153*sqrt(10659)*sh_8_6*x
    sh_9_13 = (1/153)*sqrt(12597)*sh_8_11*z + (1/153)*sqrt(20995)*sh_8_12*y - 1/153*sqrt(1615)*sh_8_13*z - 1/153*sqrt(1615)*sh_8_3*x - 1/153*sqrt(12597)*sh_8_5*x
    sh_9_14 = (1/306)*sqrt(58786)*sh_8_12*z + (2/153)*sqrt(4522)*sh_8_13*y - 1/153*sqrt(969)*sh_8_14*z - 1/153*sqrt(969)*sh_8_2*x - 1/306*sqrt(58786)*sh_8_4*x
    sh_9_15 = -1/306*sqrt(1938)*sh_8_1*x + (1/306)*sqrt(67830)*sh_8_13*z + (1/51)*sqrt(1615)*sh_8_14*y - 1/306*sqrt(1938)*sh_8_15*z - 1/306*sqrt(67830)*sh_8_3*x
    sh_9_16 = -1/306*sqrt(646)*sh_8_0*x + (2/153)*sqrt(4845)*sh_8_14*z + (4/153)*sqrt(646)*sh_8_15*y - 1/306*sqrt(646)*sh_8_16*z - 2/153*sqrt(4845)*sh_8_2*x
    sh_9_17 = (1/9)*sqrt(19)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)
    sh_9_18 = (1/6)*sqrt(38)*(-sh_8_0*x + sh_8_16*z)
    sh_10_0 = (1/10)*sqrt(105)*(sh_9_0*z + sh_9_18*x)
    sh_10_1 = (1/10)*sqrt(21)*sh_9_0*y + (3/20)*sqrt(42)*sh_9_1*z + (3/20)*sqrt(42)*sh_9_17*x
    sh_10_2 = -1/380*sqrt(798)*sh_9_0*z + (3/95)*sqrt(399)*sh_9_1*y + (3/380)*sqrt(13566)*sh_9_16*x + (1/380)*sqrt(798)*sh_9_18*x + (3/380)*sqrt(13566)*sh_9_2*z
    sh_10_3 = -3/380*sqrt(266)*sh_9_1*z + (1/95)*sqrt(6783)*sh_9_15*x + (3/380)*sqrt(266)*sh_9_17*x + (3/190)*sqrt(2261)*sh_9_2*y + (1/95)*sqrt(6783)*sh_9_3*z
    sh_10_4 = (3/95)*sqrt(665)*sh_9_14*x + (3/190)*sqrt(133)*sh_9_16*x - 3/190*sqrt(133)*sh_9_2*z + (4/95)*sqrt(399)*sh_9_3*y + (3/95)*sqrt(665)*sh_9_4*z
    sh_10_5 = (21/380)*sqrt(190)*sh_9_13*x + (1/190)*sqrt(1995)*sh_9_15*x - 1/190*sqrt(1995)*sh_9_3*z + (3/38)*sqrt(133)*sh_9_4*y + (21/380)*sqrt(190)*sh_9_5*z
    sh_10_6 = (7/380)*sqrt(1482)*sh_9_12*x + (3/380)*sqrt(1330)*sh_9_14*x - 3/380*sqrt(1330)*sh_9_4*z + (21/95)*sqrt(19)*sh_9_5*y + (7/380)*sqrt(1482)*sh_9_6*z
    sh_10_7 = (3/190)*sqrt(1729)*sh_9_11*x + (21/380)*sqrt(38)*sh_9_13*x - 21/380*sqrt(38)*sh_9_5*z + (7/190)*sqrt(741)*sh_9_6*y + (3/190)*sqrt(1729)*sh_9_7*z
    sh_10_8 = (3/190)*sqrt(1463)*sh_9_10*x + (7/190)*sqrt(114)*sh_9_12*x - 7/190*sqrt(114)*sh_9_6*z + (6/95)*sqrt(266)*sh_9_7*y + (3/190)*sqrt(1463)*sh_9_8*z
    sh_10_9 = (3/190)*sqrt(798)*sh_9_11*x - 3/190*sqrt(798)*sh_9_7*z + (3/190)*sqrt(4389)*sh_9_8*y + (1/190)*sqrt(21945)*sh_9_9*x
    sh_10_10 = -3/190*sqrt(1995)*sh_9_10*z - 3/190*sqrt(1995)*sh_9_8*x + (1/19)*sqrt(399)*sh_9_9*y
    sh_10_11 = (3/190)*sqrt(4389)*sh_9_10*y - 3/190*sqrt(798)*sh_9_11*z - 3/190*sqrt(798)*sh_9_7*x + (1/190)*sqrt(21945)*sh_9_9*z
    sh_10_12 = (3/190)*sqrt(1463)*sh_9_10*z + (6/95)*sqrt(266)*sh_9_11*y - 7/190*sqrt(114)*sh_9_12*z - 7/190*sqrt(114)*sh_9_6*x - 3/190*sqrt(1463)*sh_9_8*x
    sh_10_13 = (3/190)*sqrt(1729)*sh_9_11*z + (7/190)*sqrt(741)*sh_9_12*y - 21/380*sqrt(38)*sh_9_13*z - 21/380*sqrt(38)*sh_9_5*x - 3/190*sqrt(1729)*sh_9_7*x
    sh_10_14 = (7/380)*sqrt(1482)*sh_9_12*z + (21/95)*sqrt(19)*sh_9_13*y - 3/380*sqrt(1330)*sh_9_14*z - 3/380*sqrt(1330)*sh_9_4*x - 7/380*sqrt(1482)*sh_9_6*x
    sh_10_15 = (21/380)*sqrt(190)*sh_9_13*z + (3/38)*sqrt(133)*sh_9_14*y - 1/190*sqrt(1995)*sh_9_15*z - 1/190*sqrt(1995)*sh_9_3*x - 21/380*sqrt(190)*sh_9_5*x
    sh_10_16 = (3/95)*sqrt(665)*sh_9_14*z + (4/95)*sqrt(399)*sh_9_15*y - 3/190*sqrt(133)*sh_9_16*z - 3/190*sqrt(133)*sh_9_2*x - 3/95*sqrt(665)*sh_9_4*x
    sh_10_17 = -3/380*sqrt(266)*sh_9_1*x + (1/95)*sqrt(6783)*sh_9_15*z + (3/190)*sqrt(2261)*sh_9_16*y - 3/380*sqrt(266)*sh_9_17*z - 1/95*sqrt(6783)*sh_9_3*x
    sh_10_18 = -1/380*sqrt(798)*sh_9_0*x + (3/380)*sqrt(13566)*sh_9_16*z + (3/95)*sqrt(399)*sh_9_17*y - 1/380*sqrt(798)*sh_9_18*z - 3/380*sqrt(13566)*sh_9_2*x
    sh_10_19 = -3/20*sqrt(42)*sh_9_1*x + (3/20)*sqrt(42)*sh_9_17*z + (1/10)*sqrt(21)*sh_9_18*y
    sh_10_20 = (1/10)*sqrt(105)*(-sh_9_0*x + sh_9_18*z)
    sh_11_0 = (1/22)*sqrt(506)*(sh_10_0*z + sh_10_20*x)
    sh_11_1 = (1/11)*sqrt(23)*sh_10_0*y + (1/11)*sqrt(115)*sh_10_1*z + (1/11)*sqrt(115)*sh_10_19*x
    sh_11_2 = -1/462*sqrt(966)*sh_10_0*z + (2/231)*sqrt(4830)*sh_10_1*y + (1/231)*sqrt(45885)*sh_10_18*x + (1/231)*sqrt(45885)*sh_10_2*z + (1/462)*sqrt(966)*sh_10_20*x
    sh_11_3 = -1/154*sqrt(322)*sh_10_1*z + (1/154)*sqrt(18354)*sh_10_17*x + (1/154)*sqrt(322)*sh_10_19*x + (1/77)*sqrt(3059)*sh_10_2*y + (1/154)*sqrt(18354)*sh_10_3*z
    sh_11_4 = (1/154)*sqrt(16422)*sh_10_16*x + (1/77)*sqrt(161)*sh_10_18*x - 1/77*sqrt(161)*sh_10_2*z + (2/77)*sqrt(966)*sh_10_3*y + (1/154)*sqrt(16422)*sh_10_4*z
    sh_11_5 = (2/231)*sqrt(8211)*sh_10_15*x + (1/231)*sqrt(2415)*sh_10_17*x - 1/231*sqrt(2415)*sh_10_3*z + (1/231)*sqrt(41055)*sh_10_4*y + (2/231)*sqrt(8211)*sh_10_5*z
    sh_11_6 = (2/77)*sqrt(805)*sh_10_14*x + (1/154)*sqrt(1610)*sh_10_16*x - 1/154*sqrt(1610)*sh_10_4*z + (4/77)*sqrt(322)*sh_10_5*y + (2/77)*sqrt(805)*sh_10_6*z
    sh_11_7 = (1/22)*sqrt(230)*sh_10_13*x + (1/22)*sqrt(46)*sh_10_15*x - 1/22*sqrt(46)*sh_10_5*z + (1/11)*sqrt(115)*sh_10_6*y + (1/22)*sqrt(230)*sh_10_7*z
    sh_11_8 = (1/66)*sqrt(1794)*sh_10_12*x + (1/33)*sqrt(138)*sh_10_14*x - 1/33*sqrt(138)*sh_10_6*z + (4/33)*sqrt(69)*sh_10_7*y + (1/66)*sqrt(1794)*sh_10_8*z
    sh_11_9 = (1/77)*sqrt(2093)*sh_10_11*x + (1/77)*sqrt(966)*sh_10_13*x - 1/77*sqrt(966)*sh_10_7*z + (1/77)*sqrt(6279)*sh_10_8*y + (1/77)*sqrt(2093)*sh_10_9*z
    sh_11_10 = (1/77)*sqrt(3542)*sh_10_10*x + (1/154)*sqrt(4830)*sh_10_12*x - 1/154*sqrt(4830)*sh_10_8*z + (2/77)*sqrt(1610)*sh_10_9*y
    sh_11_11 = (1/21)*sqrt(483)*sh_10_10*y - 1/231*sqrt(26565)*sh_10_11*z - 1/231*sqrt(26565)*sh_10_9*x
    sh_11_12 = (1/77)*sqrt(3542)*sh_10_10*z + (2/77)*sqrt(1610)*sh_10_11*y - 1/154*sqrt(4830)*sh_10_12*z - 1/154*sqrt(4830)*sh_10_8*x
    sh_11_13 = (1/77)*sqrt(2093)*sh_10_11*z + (1/77)*sqrt(6279)*sh_10_12*y - 1/77*sqrt(966)*sh_10_13*z - 1/77*sqrt(966)*sh_10_7*x - 1/77*sqrt(2093)*sh_10_9*x
    sh_11_14 = (1/66)*sqrt(1794)*sh_10_12*z + (4/33)*sqrt(69)*sh_10_13*y - 1/33*sqrt(138)*sh_10_14*z - 1/33*sqrt(138)*sh_10_6*x - 1/66*sqrt(1794)*sh_10_8*x
    sh_11_15 = (1/22)*sqrt(230)*sh_10_13*z + (1/11)*sqrt(115)*sh_10_14*y - 1/22*sqrt(46)*sh_10_15*z - 1/22*sqrt(46)*sh_10_5*x - 1/22*sqrt(230)*sh_10_7*x
    sh_11_16 = (2/77)*sqrt(805)*sh_10_14*z + (4/77)*sqrt(322)*sh_10_15*y - 1/154*sqrt(1610)*sh_10_16*z - 1/154*sqrt(1610)*sh_10_4*x - 2/77*sqrt(805)*sh_10_6*x
    sh_11_17 = (2/231)*sqrt(8211)*sh_10_15*z + (1/231)*sqrt(41055)*sh_10_16*y - 1/231*sqrt(2415)*sh_10_17*z - 1/231*sqrt(2415)*sh_10_3*x - 2/231*sqrt(8211)*sh_10_5*x
    sh_11_18 = (1/154)*sqrt(16422)*sh_10_16*z + (2/77)*sqrt(966)*sh_10_17*y - 1/77*sqrt(161)*sh_10_18*z - 1/77*sqrt(161)*sh_10_2*x - 1/154*sqrt(16422)*sh_10_4*x
    sh_11_19 = -1/154*sqrt(322)*sh_10_1*x + (1/154)*sqrt(18354)*sh_10_17*z + (1/77)*sqrt(3059)*sh_10_18*y - 1/154*sqrt(322)*sh_10_19*z - 1/154*sqrt(18354)*sh_10_3*x
    sh_11_20 = -1/462*sqrt(966)*sh_10_0*x + (1/231)*sqrt(45885)*sh_10_18*z + (2/231)*sqrt(4830)*sh_10_19*y - 1/231*sqrt(45885)*sh_10_2*x - 1/462*sqrt(966)*sh_10_20*z
    sh_11_21 = -1/11*sqrt(115)*sh_10_1*x + (1/11)*sqrt(115)*sh_10_19*z + (1/11)*sqrt(23)*sh_10_20*y
    sh_11_22 = (1/22)*sqrt(506)*(-sh_10_0*x + sh_10_20*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
        sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22
    ], dim=-1)


def _sph_lmax_12(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = sqrt(3)*x
    sh_1_1 = sqrt(3)*y
    sh_1_2 = sqrt(3)*z
    sh_2_0 = (1/2)*sqrt(5)*(sh_1_0*z + sh_1_2*x)
    sh_2_1 = (1/2)*sqrt(5)*(sh_1_0*y + sh_1_1*x)
    sh_2_2 = (1/6)*sqrt(15)*(-sh_1_0*x + 2*sh_1_1*y - sh_1_2*z)
    sh_2_3 = (1/2)*sqrt(5)*(sh_1_1*z + sh_1_2*y)
    sh_2_4 = (1/2)*sqrt(5)*(-sh_1_0*x + sh_1_2*z)
    sh_3_0 = (1/6)*sqrt(42)*(sh_2_0*z + sh_2_4*x)
    sh_3_1 = (1/3)*sqrt(7)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)
    sh_3_2 = -1/30*sqrt(70)*sh_2_0*z + (2/15)*sqrt(70)*sh_2_1*y + (1/15)*sqrt(210)*sh_2_2*x + (1/30)*sqrt(70)*sh_2_4*x
    sh_3_3 = -1/15*sqrt(105)*sh_2_1*x + (1/5)*sqrt(35)*sh_2_2*y - 1/15*sqrt(105)*sh_2_3*z
    sh_3_4 = -1/30*sqrt(70)*sh_2_0*x + (1/15)*sqrt(210)*sh_2_2*z + (2/15)*sqrt(70)*sh_2_3*y - 1/30*sqrt(70)*sh_2_4*z
    sh_3_5 = (1/3)*sqrt(7)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)
    sh_3_6 = (1/6)*sqrt(42)*(-sh_2_0*x + sh_2_4*z)
    sh_4_0 = (3/4)*sqrt(2)*(sh_3_0*z + sh_3_6*x)
    sh_4_1 = (3/4)*sh_3_0*y + (3/8)*sqrt(6)*sh_3_1*z + (3/8)*sqrt(6)*sh_3_5*x
    sh_4_2 = -3/56*sqrt(14)*sh_3_0*z + (3/14)*sqrt(21)*sh_3_1*y + (3/56)*sqrt(210)*sh_3_2*z + (3/56)*sqrt(210)*sh_3_4*x + (3/56)*sqrt(14)*sh_3_6*x
    sh_4_3 = -3/56*sqrt(42)*sh_3_1*z + (3/28)*sqrt(105)*sh_3_2*y + (3/28)*sqrt(70)*sh_3_3*x + (3/56)*sqrt(42)*sh_3_5*x
    sh_4_4 = -3/28*sqrt(42)*sh_3_2*x + (3/7)*sqrt(7)*sh_3_3*y - 3/28*sqrt(42)*sh_3_4*z
    sh_4_5 = -3/56*sqrt(42)*sh_3_1*x + (3/28)*sqrt(70)*sh_3_3*z + (3/28)*sqrt(105)*sh_3_4*y - 3/56*sqrt(42)*sh_3_5*z
    sh_4_6 = -3/56*sqrt(14)*sh_3_0*x - 3/56*sqrt(210)*sh_3_2*x + (3/56)*sqrt(210)*sh_3_4*z + (3/14)*sqrt(21)*sh_3_5*y - 3/56*sqrt(14)*sh_3_6*z
    sh_4_7 = -3/8*sqrt(6)*sh_3_1*x + (3/8)*sqrt(6)*sh_3_5*z + (3/4)*sh_3_6*y
    sh_4_8 = (3/4)*sqrt(2)*(-sh_3_0*x + sh_3_6*z)
    sh_5_0 = (1/10)*sqrt(110)*(sh_4_0*z + sh_4_8*x)
    sh_5_1 = (1/5)*sqrt(11)*sh_4_0*y + (1/5)*sqrt(22)*sh_4_1*z + (1/5)*sqrt(22)*sh_4_7*x
    sh_5_2 = -1/30*sqrt(22)*sh_4_0*z + (4/15)*sqrt(11)*sh_4_1*y + (1/15)*sqrt(154)*sh_4_2*z + (1/15)*sqrt(154)*sh_4_6*x + (1/30)*sqrt(22)*sh_4_8*x
    sh_5_3 = -1/30*sqrt(66)*sh_4_1*z + (1/15)*sqrt(231)*sh_4_2*y + (1/30)*sqrt(462)*sh_4_3*z + (1/30)*sqrt(462)*sh_4_5*x + (1/30)*sqrt(66)*sh_4_7*x
    sh_5_4 = -1/15*sqrt(33)*sh_4_2*z + (2/15)*sqrt(66)*sh_4_3*y + (1/15)*sqrt(165)*sh_4_4*x + (1/15)*sqrt(33)*sh_4_6*x
    sh_5_5 = -1/15*sqrt(110)*sh_4_3*x + (1/3)*sqrt(11)*sh_4_4*y - 1/15*sqrt(110)*sh_4_5*z
    sh_5_6 = -1/15*sqrt(33)*sh_4_2*x + (1/15)*sqrt(165)*sh_4_4*z + (2/15)*sqrt(66)*sh_4_5*y - 1/15*sqrt(33)*sh_4_6*z
    sh_5_7 = -1/30*sqrt(66)*sh_4_1*x - 1/30*sqrt(462)*sh_4_3*x + (1/30)*sqrt(462)*sh_4_5*z + (1/15)*sqrt(231)*sh_4_6*y - 1/30*sqrt(66)*sh_4_7*z
    sh_5_8 = -1/30*sqrt(22)*sh_4_0*x - 1/15*sqrt(154)*sh_4_2*x + (1/15)*sqrt(154)*sh_4_6*z + (4/15)*sqrt(11)*sh_4_7*y - 1/30*sqrt(22)*sh_4_8*z
    sh_5_9 = -1/5*sqrt(22)*sh_4_1*x + (1/5)*sqrt(22)*sh_4_7*z + (1/5)*sqrt(11)*sh_4_8*y
    sh_5_10 = (1/10)*sqrt(110)*(-sh_4_0*x + sh_4_8*z)
    sh_6_0 = (1/6)*sqrt(39)*(sh_5_0*z + sh_5_10*x)
    sh_6_1 = (1/6)*sqrt(13)*sh_5_0*y + (1/12)*sqrt(130)*sh_5_1*z + (1/12)*sqrt(130)*sh_5_9*x
    sh_6_2 = -1/132*sqrt(286)*sh_5_0*z + (1/33)*sqrt(715)*sh_5_1*y + (1/132)*sqrt(286)*sh_5_10*x + (1/44)*sqrt(1430)*sh_5_2*z + (1/44)*sqrt(1430)*sh_5_8*x
    sh_6_3 = -1/132*sqrt(858)*sh_5_1*z + (1/22)*sqrt(429)*sh_5_2*y + (1/22)*sqrt(286)*sh_5_3*z + (1/22)*sqrt(286)*sh_5_7*x + (1/132)*sqrt(858)*sh_5_9*x
    sh_6_4 = -1/66*sqrt(429)*sh_5_2*z + (2/33)*sqrt(286)*sh_5_3*y + (1/66)*sqrt(2002)*sh_5_4*z + (1/66)*sqrt(2002)*sh_5_6*x + (1/66)*sqrt(429)*sh_5_8*x
    sh_6_5 = -1/66*sqrt(715)*sh_5_3*z + (1/66)*sqrt(5005)*sh_5_4*y + (1/66)*sqrt(3003)*sh_5_5*x + (1/66)*sqrt(715)*sh_5_7*x
    sh_6_6 = -1/66*sqrt(2145)*sh_5_4*x + (1/11)*sqrt(143)*sh_5_5*y - 1/66*sqrt(2145)*sh_5_6*z
    sh_6_7 = -1/66*sqrt(715)*sh_5_3*x + (1/66)*sqrt(3003)*sh_5_5*z + (1/66)*sqrt(5005)*sh_5_6*y - 1/66*sqrt(715)*sh_5_7*z
    sh_6_8 = -1/66*sqrt(429)*sh_5_2*x - 1/66*sqrt(2002)*sh_5_4*x + (1/66)*sqrt(2002)*sh_5_6*z + (2/33)*sqrt(286)*sh_5_7*y - 1/66*sqrt(429)*sh_5_8*z
    sh_6_9 = -1/132*sqrt(858)*sh_5_1*x - 1/22*sqrt(286)*sh_5_3*x + (1/22)*sqrt(286)*sh_5_7*z + (1/22)*sqrt(429)*sh_5_8*y - 1/132*sqrt(858)*sh_5_9*z
    sh_6_10 = -1/132*sqrt(286)*sh_5_0*x - 1/132*sqrt(286)*sh_5_10*z - 1/44*sqrt(1430)*sh_5_2*x + (1/44)*sqrt(1430)*sh_5_8*z + (1/33)*sqrt(715)*sh_5_9*y
    sh_6_11 = -1/12*sqrt(130)*sh_5_1*x + (1/6)*sqrt(13)*sh_5_10*y + (1/12)*sqrt(130)*sh_5_9*z
    sh_6_12 = (1/6)*sqrt(39)*(-sh_5_0*x + sh_5_10*z)
    sh_7_0 = (1/14)*sqrt(210)*(sh_6_0*z + sh_6_12*x)
    sh_7_1 = (1/7)*sqrt(15)*sh_6_0*y + (3/7)*sqrt(5)*sh_6_1*z + (3/7)*sqrt(5)*sh_6_11*x
    sh_7_2 = -1/182*sqrt(390)*sh_6_0*z + (6/91)*sqrt(130)*sh_6_1*y + (3/91)*sqrt(715)*sh_6_10*x + (1/182)*sqrt(390)*sh_6_12*x + (3/91)*sqrt(715)*sh_6_2*z
    sh_7_3 = -3/182*sqrt(130)*sh_6_1*z + (3/182)*sqrt(130)*sh_6_11*x + (3/91)*sqrt(715)*sh_6_2*y + (5/182)*sqrt(858)*sh_6_3*z + (5/182)*sqrt(858)*sh_6_9*x
    sh_7_4 = (3/91)*sqrt(65)*sh_6_10*x - 3/91*sqrt(65)*sh_6_2*z + (10/91)*sqrt(78)*sh_6_3*y + (15/182)*sqrt(78)*sh_6_4*z + (15/182)*sqrt(78)*sh_6_8*x
    sh_7_5 = -5/91*sqrt(39)*sh_6_3*z + (15/91)*sqrt(39)*sh_6_4*y + (3/91)*sqrt(390)*sh_6_5*z + (3/91)*sqrt(390)*sh_6_7*x + (5/91)*sqrt(39)*sh_6_9*x
    sh_7_6 = -15/182*sqrt(26)*sh_6_4*z + (12/91)*sqrt(65)*sh_6_5*y + (2/91)*sqrt(1365)*sh_6_6*x + (15/182)*sqrt(26)*sh_6_8*x
    sh_7_7 = -3/91*sqrt(455)*sh_6_5*x + (1/13)*sqrt(195)*sh_6_6*y - 3/91*sqrt(455)*sh_6_7*z
    sh_7_8 = -15/182*sqrt(26)*sh_6_4*x + (2/91)*sqrt(1365)*sh_6_6*z + (12/91)*sqrt(65)*sh_6_7*y - 15/182*sqrt(26)*sh_6_8*z
    sh_7_9 = -5/91*sqrt(39)*sh_6_3*x - 3/91*sqrt(390)*sh_6_5*x + (3/91)*sqrt(390)*sh_6_7*z + (15/91)*sqrt(39)*sh_6_8*y - 5/91*sqrt(39)*sh_6_9*z
    sh_7_10 = -3/91*sqrt(65)*sh_6_10*z - 3/91*sqrt(65)*sh_6_2*x - 15/182*sqrt(78)*sh_6_4*x + (15/182)*sqrt(78)*sh_6_8*z + (10/91)*sqrt(78)*sh_6_9*y
    sh_7_11 = -3/182*sqrt(130)*sh_6_1*x + (3/91)*sqrt(715)*sh_6_10*y - 3/182*sqrt(130)*sh_6_11*z - 5/182*sqrt(858)*sh_6_3*x + (5/182)*sqrt(858)*sh_6_9*z
    sh_7_12 = -1/182*sqrt(390)*sh_6_0*x + (3/91)*sqrt(715)*sh_6_10*z + (6/91)*sqrt(130)*sh_6_11*y - 1/182*sqrt(390)*sh_6_12*z - 3/91*sqrt(715)*sh_6_2*x
    sh_7_13 = -3/7*sqrt(5)*sh_6_1*x + (3/7)*sqrt(5)*sh_6_11*z + (1/7)*sqrt(15)*sh_6_12*y
    sh_7_14 = (1/14)*sqrt(210)*(-sh_6_0*x + sh_6_12*z)
    sh_8_0 = (1/4)*sqrt(17)*(sh_7_0*z + sh_7_14*x)
    sh_8_1 = (1/8)*sqrt(17)*sh_7_0*y + (1/16)*sqrt(238)*sh_7_1*z + (1/16)*sqrt(238)*sh_7_13*x
    sh_8_2 = -1/240*sqrt(510)*sh_7_0*z + (1/60)*sqrt(1785)*sh_7_1*y + (1/240)*sqrt(46410)*sh_7_12*x + (1/240)*sqrt(510)*sh_7_14*x + (1/240)*sqrt(46410)*sh_7_2*z
    sh_8_3 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)
    sh_8_4 = (1/40)*sqrt(935)*sh_7_10*x + (1/40)*sqrt(85)*sh_7_12*x - 1/40*sqrt(85)*sh_7_2*z + (1/10)*sqrt(85)*sh_7_3*y + (1/40)*sqrt(935)*sh_7_4*z
    sh_8_5 = (1/48)*sqrt(2)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)
    sh_8_6 = (1/16)*sqrt(34)*sh_7_10*x - 1/16*sqrt(34)*sh_7_4*z + (1/4)*sqrt(17)*sh_7_5*y + (1/16)*sqrt(102)*sh_7_6*z + (1/16)*sqrt(102)*sh_7_8*x
    sh_8_7 = -1/80*sqrt(1190)*sh_7_5*z + (1/40)*sqrt(1785)*sh_7_6*y + (1/20)*sqrt(255)*sh_7_7*x + (1/80)*sqrt(1190)*sh_7_9*x
    sh_8_8 = -1/60*sqrt(1785)*sh_7_6*x + (1/15)*sqrt(255)*sh_7_7*y - 1/60*sqrt(1785)*sh_7_8*z
    sh_8_9 = -1/80*sqrt(1190)*sh_7_5*x + (1/20)*sqrt(255)*sh_7_7*z + (1/40)*sqrt(1785)*sh_7_8*y - 1/80*sqrt(1190)*sh_7_9*z
    sh_8_10 = -1/16*sqrt(34)*sh_7_10*z - 1/16*sqrt(34)*sh_7_4*x - 1/16*sqrt(102)*sh_7_6*x + (1/16)*sqrt(102)*sh_7_8*z + (1/4)*sqrt(17)*sh_7_9*y
    sh_8_11 = (1/48)*sqrt(2)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)
    sh_8_12 = (1/40)*sqrt(935)*sh_7_10*z + (1/10)*sqrt(85)*sh_7_11*y - 1/40*sqrt(85)*sh_7_12*z - 1/40*sqrt(85)*sh_7_2*x - 1/40*sqrt(935)*sh_7_4*x
    sh_8_13 = (1/80)*sqrt(2)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)
    sh_8_14 = -1/240*sqrt(510)*sh_7_0*x + (1/240)*sqrt(46410)*sh_7_12*z + (1/60)*sqrt(1785)*sh_7_13*y - 1/240*sqrt(510)*sh_7_14*z - 1/240*sqrt(46410)*sh_7_2*x
    sh_8_15 = -1/16*sqrt(238)*sh_7_1*x + (1/16)*sqrt(238)*sh_7_13*z + (1/8)*sqrt(17)*sh_7_14*y
    sh_8_16 = (1/4)*sqrt(17)*(-sh_7_0*x + sh_7_14*z)
    sh_9_0 = (1/6)*sqrt(38)*(sh_8_0*z + sh_8_16*x)
    sh_9_1 = (1/9)*sqrt(19)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)
    sh_9_2 = -1/306*sqrt(646)*sh_8_0*z + (4/153)*sqrt(646)*sh_8_1*y + (2/153)*sqrt(4845)*sh_8_14*x + (1/306)*sqrt(646)*sh_8_16*x + (2/153)*sqrt(4845)*sh_8_2*z
    sh_9_3 = -1/306*sqrt(1938)*sh_8_1*z + (1/306)*sqrt(67830)*sh_8_13*x + (1/306)*sqrt(1938)*sh_8_15*x + (1/51)*sqrt(1615)*sh_8_2*y + (1/306)*sqrt(67830)*sh_8_3*z
    sh_9_4 = (1/306)*sqrt(58786)*sh_8_12*x + (1/153)*sqrt(969)*sh_8_14*x - 1/153*sqrt(969)*sh_8_2*z + (2/153)*sqrt(4522)*sh_8_3*y + (1/306)*sqrt(58786)*sh_8_4*z
    sh_9_5 = (1/153)*sqrt(12597)*sh_8_11*x + (1/153)*sqrt(1615)*sh_8_13*x - 1/153*sqrt(1615)*sh_8_3*z + (1/153)*sqrt(20995)*sh_8_4*y + (1/153)*sqrt(12597)*sh_8_5*z
    sh_9_6 = (1/153)*sqrt(10659)*sh_8_10*x + (1/306)*sqrt(9690)*sh_8_12*x - 1/306*sqrt(9690)*sh_8_4*z + (2/51)*sqrt(646)*sh_8_5*y + (1/153)*sqrt(10659)*sh_8_6*z
    sh_9_7 = (1/306)*sqrt(13566)*sh_8_11*x - 1/306*sqrt(13566)*sh_8_5*z + (1/153)*sqrt(24871)*sh_8_6*y + (1/306)*sqrt(35530)*sh_8_7*z + (1/306)*sqrt(35530)*sh_8_9*x
    sh_9_8 = (1/153)*sqrt(4522)*sh_8_10*x - 1/153*sqrt(4522)*sh_8_6*z + (4/153)*sqrt(1615)*sh_8_7*y + (1/51)*sqrt(1615)*sh_8_8*x
    sh_9_9 = (1/51)*sqrt(323)*(-2*sh_8_7*x + 3*sh_8_8*y - 2*sh_8_9*z)
    sh_9_10 = -1/153*sqrt(4522)*sh_8_10*z - 1/153*sqrt(4522)*sh_8_6*x + (1/51)*sqrt(1615)*sh_8_8*z + (4/153)*sqrt(1615)*sh_8_9*y
    sh_9_11 = (1/153)*sqrt(24871)*sh_8_10*y - 1/306*sqrt(13566)*sh_8_11*z - 1/306*sqrt(13566)*sh_8_5*x - 1/306*sqrt(35530)*sh_8_7*x + (1/306)*sqrt(35530)*sh_8_9*z
    sh_9_12 = (1/153)*sqrt(10659)*sh_8_10*z + (2/51)*sqrt(646)*sh_8_11*y - 1/306*sqrt(9690)*sh_8_12*z - 1/306*sqrt(9690)*sh_8_4*x - 1/153*sqrt(10659)*sh_8_6*x
    sh_9_13 = (1/153)*sqrt(12597)*sh_8_11*z + (1/153)*sqrt(20995)*sh_8_12*y - 1/153*sqrt(1615)*sh_8_13*z - 1/153*sqrt(1615)*sh_8_3*x - 1/153*sqrt(12597)*sh_8_5*x
    sh_9_14 = (1/306)*sqrt(58786)*sh_8_12*z + (2/153)*sqrt(4522)*sh_8_13*y - 1/153*sqrt(969)*sh_8_14*z - 1/153*sqrt(969)*sh_8_2*x - 1/306*sqrt(58786)*sh_8_4*x
    sh_9_15 = -1/306*sqrt(1938)*sh_8_1*x + (1/306)*sqrt(67830)*sh_8_13*z + (1/51)*sqrt(1615)*sh_8_14*y - 1/306*sqrt(1938)*sh_8_15*z - 1/306*sqrt(67830)*sh_8_3*x
    sh_9_16 = -1/306*sqrt(646)*sh_8_0*x + (2/153)*sqrt(4845)*sh_8_14*z + (4/153)*sqrt(646)*sh_8_15*y - 1/306*sqrt(646)*sh_8_16*z - 2/153*sqrt(4845)*sh_8_2*x
    sh_9_17 = (1/9)*sqrt(19)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)
    sh_9_18 = (1/6)*sqrt(38)*(-sh_8_0*x + sh_8_16*z)
    sh_10_0 = (1/10)*sqrt(105)*(sh_9_0*z + sh_9_18*x)
    sh_10_1 = (1/10)*sqrt(21)*sh_9_0*y + (3/20)*sqrt(42)*sh_9_1*z + (3/20)*sqrt(42)*sh_9_17*x
    sh_10_2 = -1/380*sqrt(798)*sh_9_0*z + (3/95)*sqrt(399)*sh_9_1*y + (3/380)*sqrt(13566)*sh_9_16*x + (1/380)*sqrt(798)*sh_9_18*x + (3/380)*sqrt(13566)*sh_9_2*z
    sh_10_3 = -3/380*sqrt(266)*sh_9_1*z + (1/95)*sqrt(6783)*sh_9_15*x + (3/380)*sqrt(266)*sh_9_17*x + (3/190)*sqrt(2261)*sh_9_2*y + (1/95)*sqrt(6783)*sh_9_3*z
    sh_10_4 = (3/95)*sqrt(665)*sh_9_14*x + (3/190)*sqrt(133)*sh_9_16*x - 3/190*sqrt(133)*sh_9_2*z + (4/95)*sqrt(399)*sh_9_3*y + (3/95)*sqrt(665)*sh_9_4*z
    sh_10_5 = (21/380)*sqrt(190)*sh_9_13*x + (1/190)*sqrt(1995)*sh_9_15*x - 1/190*sqrt(1995)*sh_9_3*z + (3/38)*sqrt(133)*sh_9_4*y + (21/380)*sqrt(190)*sh_9_5*z
    sh_10_6 = (7/380)*sqrt(1482)*sh_9_12*x + (3/380)*sqrt(1330)*sh_9_14*x - 3/380*sqrt(1330)*sh_9_4*z + (21/95)*sqrt(19)*sh_9_5*y + (7/380)*sqrt(1482)*sh_9_6*z
    sh_10_7 = (3/190)*sqrt(1729)*sh_9_11*x + (21/380)*sqrt(38)*sh_9_13*x - 21/380*sqrt(38)*sh_9_5*z + (7/190)*sqrt(741)*sh_9_6*y + (3/190)*sqrt(1729)*sh_9_7*z
    sh_10_8 = (3/190)*sqrt(1463)*sh_9_10*x + (7/190)*sqrt(114)*sh_9_12*x - 7/190*sqrt(114)*sh_9_6*z + (6/95)*sqrt(266)*sh_9_7*y + (3/190)*sqrt(1463)*sh_9_8*z
    sh_10_9 = (3/190)*sqrt(798)*sh_9_11*x - 3/190*sqrt(798)*sh_9_7*z + (3/190)*sqrt(4389)*sh_9_8*y + (1/190)*sqrt(21945)*sh_9_9*x
    sh_10_10 = -3/190*sqrt(1995)*sh_9_10*z - 3/190*sqrt(1995)*sh_9_8*x + (1/19)*sqrt(399)*sh_9_9*y
    sh_10_11 = (3/190)*sqrt(4389)*sh_9_10*y - 3/190*sqrt(798)*sh_9_11*z - 3/190*sqrt(798)*sh_9_7*x + (1/190)*sqrt(21945)*sh_9_9*z
    sh_10_12 = (3/190)*sqrt(1463)*sh_9_10*z + (6/95)*sqrt(266)*sh_9_11*y - 7/190*sqrt(114)*sh_9_12*z - 7/190*sqrt(114)*sh_9_6*x - 3/190*sqrt(1463)*sh_9_8*x
    sh_10_13 = (3/190)*sqrt(1729)*sh_9_11*z + (7/190)*sqrt(741)*sh_9_12*y - 21/380*sqrt(38)*sh_9_13*z - 21/380*sqrt(38)*sh_9_5*x - 3/190*sqrt(1729)*sh_9_7*x
    sh_10_14 = (7/380)*sqrt(1482)*sh_9_12*z + (21/95)*sqrt(19)*sh_9_13*y - 3/380*sqrt(1330)*sh_9_14*z - 3/380*sqrt(1330)*sh_9_4*x - 7/380*sqrt(1482)*sh_9_6*x
    sh_10_15 = (21/380)*sqrt(190)*sh_9_13*z + (3/38)*sqrt(133)*sh_9_14*y - 1/190*sqrt(1995)*sh_9_15*z - 1/190*sqrt(1995)*sh_9_3*x - 21/380*sqrt(190)*sh_9_5*x
    sh_10_16 = (3/95)*sqrt(665)*sh_9_14*z + (4/95)*sqrt(399)*sh_9_15*y - 3/190*sqrt(133)*sh_9_16*z - 3/190*sqrt(133)*sh_9_2*x - 3/95*sqrt(665)*sh_9_4*x
    sh_10_17 = -3/380*sqrt(266)*sh_9_1*x + (1/95)*sqrt(6783)*sh_9_15*z + (3/190)*sqrt(2261)*sh_9_16*y - 3/380*sqrt(266)*sh_9_17*z - 1/95*sqrt(6783)*sh_9_3*x
    sh_10_18 = -1/380*sqrt(798)*sh_9_0*x + (3/380)*sqrt(13566)*sh_9_16*z + (3/95)*sqrt(399)*sh_9_17*y - 1/380*sqrt(798)*sh_9_18*z - 3/380*sqrt(13566)*sh_9_2*x
    sh_10_19 = -3/20*sqrt(42)*sh_9_1*x + (3/20)*sqrt(42)*sh_9_17*z + (1/10)*sqrt(21)*sh_9_18*y
    sh_10_20 = (1/10)*sqrt(105)*(-sh_9_0*x + sh_9_18*z)
    sh_11_0 = (1/22)*sqrt(506)*(sh_10_0*z + sh_10_20*x)
    sh_11_1 = (1/11)*sqrt(23)*sh_10_0*y + (1/11)*sqrt(115)*sh_10_1*z + (1/11)*sqrt(115)*sh_10_19*x
    sh_11_2 = -1/462*sqrt(966)*sh_10_0*z + (2/231)*sqrt(4830)*sh_10_1*y + (1/231)*sqrt(45885)*sh_10_18*x + (1/231)*sqrt(45885)*sh_10_2*z + (1/462)*sqrt(966)*sh_10_20*x
    sh_11_3 = -1/154*sqrt(322)*sh_10_1*z + (1/154)*sqrt(18354)*sh_10_17*x + (1/154)*sqrt(322)*sh_10_19*x + (1/77)*sqrt(3059)*sh_10_2*y + (1/154)*sqrt(18354)*sh_10_3*z
    sh_11_4 = (1/154)*sqrt(16422)*sh_10_16*x + (1/77)*sqrt(161)*sh_10_18*x - 1/77*sqrt(161)*sh_10_2*z + (2/77)*sqrt(966)*sh_10_3*y + (1/154)*sqrt(16422)*sh_10_4*z
    sh_11_5 = (2/231)*sqrt(8211)*sh_10_15*x + (1/231)*sqrt(2415)*sh_10_17*x - 1/231*sqrt(2415)*sh_10_3*z + (1/231)*sqrt(41055)*sh_10_4*y + (2/231)*sqrt(8211)*sh_10_5*z
    sh_11_6 = (2/77)*sqrt(805)*sh_10_14*x + (1/154)*sqrt(1610)*sh_10_16*x - 1/154*sqrt(1610)*sh_10_4*z + (4/77)*sqrt(322)*sh_10_5*y + (2/77)*sqrt(805)*sh_10_6*z
    sh_11_7 = (1/22)*sqrt(230)*sh_10_13*x + (1/22)*sqrt(46)*sh_10_15*x - 1/22*sqrt(46)*sh_10_5*z + (1/11)*sqrt(115)*sh_10_6*y + (1/22)*sqrt(230)*sh_10_7*z
    sh_11_8 = (1/66)*sqrt(1794)*sh_10_12*x + (1/33)*sqrt(138)*sh_10_14*x - 1/33*sqrt(138)*sh_10_6*z + (4/33)*sqrt(69)*sh_10_7*y + (1/66)*sqrt(1794)*sh_10_8*z
    sh_11_9 = (1/77)*sqrt(2093)*sh_10_11*x + (1/77)*sqrt(966)*sh_10_13*x - 1/77*sqrt(966)*sh_10_7*z + (1/77)*sqrt(6279)*sh_10_8*y + (1/77)*sqrt(2093)*sh_10_9*z
    sh_11_10 = (1/77)*sqrt(3542)*sh_10_10*x + (1/154)*sqrt(4830)*sh_10_12*x - 1/154*sqrt(4830)*sh_10_8*z + (2/77)*sqrt(1610)*sh_10_9*y
    sh_11_11 = (1/21)*sqrt(483)*sh_10_10*y - 1/231*sqrt(26565)*sh_10_11*z - 1/231*sqrt(26565)*sh_10_9*x
    sh_11_12 = (1/77)*sqrt(3542)*sh_10_10*z + (2/77)*sqrt(1610)*sh_10_11*y - 1/154*sqrt(4830)*sh_10_12*z - 1/154*sqrt(4830)*sh_10_8*x
    sh_11_13 = (1/77)*sqrt(2093)*sh_10_11*z + (1/77)*sqrt(6279)*sh_10_12*y - 1/77*sqrt(966)*sh_10_13*z - 1/77*sqrt(966)*sh_10_7*x - 1/77*sqrt(2093)*sh_10_9*x
    sh_11_14 = (1/66)*sqrt(1794)*sh_10_12*z + (4/33)*sqrt(69)*sh_10_13*y - 1/33*sqrt(138)*sh_10_14*z - 1/33*sqrt(138)*sh_10_6*x - 1/66*sqrt(1794)*sh_10_8*x
    sh_11_15 = (1/22)*sqrt(230)*sh_10_13*z + (1/11)*sqrt(115)*sh_10_14*y - 1/22*sqrt(46)*sh_10_15*z - 1/22*sqrt(46)*sh_10_5*x - 1/22*sqrt(230)*sh_10_7*x
    sh_11_16 = (2/77)*sqrt(805)*sh_10_14*z + (4/77)*sqrt(322)*sh_10_15*y - 1/154*sqrt(1610)*sh_10_16*z - 1/154*sqrt(1610)*sh_10_4*x - 2/77*sqrt(805)*sh_10_6*x
    sh_11_17 = (2/231)*sqrt(8211)*sh_10_15*z + (1/231)*sqrt(41055)*sh_10_16*y - 1/231*sqrt(2415)*sh_10_17*z - 1/231*sqrt(2415)*sh_10_3*x - 2/231*sqrt(8211)*sh_10_5*x
    sh_11_18 = (1/154)*sqrt(16422)*sh_10_16*z + (2/77)*sqrt(966)*sh_10_17*y - 1/77*sqrt(161)*sh_10_18*z - 1/77*sqrt(161)*sh_10_2*x - 1/154*sqrt(16422)*sh_10_4*x
    sh_11_19 = -1/154*sqrt(322)*sh_10_1*x + (1/154)*sqrt(18354)*sh_10_17*z + (1/77)*sqrt(3059)*sh_10_18*y - 1/154*sqrt(322)*sh_10_19*z - 1/154*sqrt(18354)*sh_10_3*x
    sh_11_20 = -1/462*sqrt(966)*sh_10_0*x + (1/231)*sqrt(45885)*sh_10_18*z + (2/231)*sqrt(4830)*sh_10_19*y - 1/231*sqrt(45885)*sh_10_2*x - 1/462*sqrt(966)*sh_10_20*z
    sh_11_21 = -1/11*sqrt(115)*sh_10_1*x + (1/11)*sqrt(115)*sh_10_19*z + (1/11)*sqrt(23)*sh_10_20*y
    sh_11_22 = (1/22)*sqrt(506)*(-sh_10_0*x + sh_10_20*z)
    sh_12_0 = (5/12)*sqrt(6)*(sh_11_0*z + sh_11_22*x)
    sh_12_1 = (5/12)*sh_11_0*y + (5/24)*sqrt(22)*sh_11_1*z + (5/24)*sqrt(22)*sh_11_21*x
    sh_12_2 = -5/552*sqrt(46)*sh_11_0*z + (5/138)*sqrt(253)*sh_11_1*y + (5/552)*sqrt(10626)*sh_11_2*z + (5/552)*sqrt(10626)*sh_11_20*x + (5/552)*sqrt(46)*sh_11_22*x
    sh_12_3 = -5/552*sqrt(138)*sh_11_1*z + (5/276)*sqrt(2415)*sh_11_19*x + (5/92)*sqrt(161)*sh_11_2*y + (5/552)*sqrt(138)*sh_11_21*x + (5/276)*sqrt(2415)*sh_11_3*z
    sh_12_4 = (5/276)*sqrt(2185)*sh_11_18*x - 5/276*sqrt(69)*sh_11_2*z + (5/276)*sqrt(69)*sh_11_20*x + (5/69)*sqrt(115)*sh_11_3*y + (5/276)*sqrt(2185)*sh_11_4*z
    sh_12_5 = (5/184)*sqrt(874)*sh_11_17*x + (5/276)*sqrt(115)*sh_11_19*x - 5/276*sqrt(115)*sh_11_3*z + (5/276)*sqrt(2185)*sh_11_4*y + (5/184)*sqrt(874)*sh_11_5*z
    sh_12_6 = (5/552)*sqrt(3)*(sqrt(2346)*sh_11_16*x + sqrt(230)*sh_11_18*x - sqrt(230)*sh_11_4*z + 12*sqrt(23)*sh_11_5*y + sqrt(2346)*sh_11_6*z)
    sh_12_7 = (5/138)*sqrt(391)*sh_11_15*x + (5/552)*sqrt(966)*sh_11_17*x - 5/552*sqrt(966)*sh_11_5*z + (5/276)*sqrt(2737)*sh_11_6*y + (5/138)*sqrt(391)*sh_11_7*z
    sh_12_8 = (5/138)*sqrt(345)*sh_11_14*x + (5/276)*sqrt(322)*sh_11_16*x - 5/276*sqrt(322)*sh_11_6*z + (10/69)*sqrt(46)*sh_11_7*y + (5/138)*sqrt(345)*sh_11_8*z
    sh_12_9 = (5/552)*sqrt(4830)*sh_11_13*x + (5/92)*sqrt(46)*sh_11_15*x - 5/92*sqrt(46)*sh_11_7*z + (5/92)*sqrt(345)*sh_11_8*y + (5/552)*sqrt(4830)*sh_11_9*z
    sh_12_10 = (5/552)*sqrt(4186)*sh_11_10*z + (5/552)*sqrt(4186)*sh_11_12*x + (5/184)*sqrt(230)*sh_11_14*x - 5/184*sqrt(230)*sh_11_8*z + (5/138)*sqrt(805)*sh_11_9*y
    sh_12_11 = (5/276)*sqrt(3289)*sh_11_10*y + (5/276)*sqrt(1794)*sh_11_11*x + (5/552)*sqrt(2530)*sh_11_13*x - 5/552*sqrt(2530)*sh_11_9*z
    sh_12_12 = -5/276*sqrt(1518)*sh_11_10*x + (5/23)*sqrt(23)*sh_11_11*y - 5/276*sqrt(1518)*sh_11_12*z
    sh_12_13 = (5/276)*sqrt(1794)*sh_11_11*z + (5/276)*sqrt(3289)*sh_11_12*y - 5/552*sqrt(2530)*sh_11_13*z - 5/552*sqrt(2530)*sh_11_9*x
    sh_12_14 = -5/552*sqrt(4186)*sh_11_10*x + (5/552)*sqrt(4186)*sh_11_12*z + (5/138)*sqrt(805)*sh_11_13*y - 5/184*sqrt(230)*sh_11_14*z - 5/184*sqrt(230)*sh_11_8*x
    sh_12_15 = (5/552)*sqrt(4830)*sh_11_13*z + (5/92)*sqrt(345)*sh_11_14*y - 5/92*sqrt(46)*sh_11_15*z - 5/92*sqrt(46)*sh_11_7*x - 5/552*sqrt(4830)*sh_11_9*x
    sh_12_16 = (5/138)*sqrt(345)*sh_11_14*z + (10/69)*sqrt(46)*sh_11_15*y - 5/276*sqrt(322)*sh_11_16*z - 5/276*sqrt(322)*sh_11_6*x - 5/138*sqrt(345)*sh_11_8*x
    sh_12_17 = (5/138)*sqrt(391)*sh_11_15*z + (5/276)*sqrt(2737)*sh_11_16*y - 5/552*sqrt(966)*sh_11_17*z - 5/552*sqrt(966)*sh_11_5*x - 5/138*sqrt(391)*sh_11_7*x
    sh_12_18 = (5/552)*sqrt(3)*(sqrt(2346)*sh_11_16*z + 12*sqrt(23)*sh_11_17*y - sqrt(230)*sh_11_18*z - sqrt(230)*sh_11_4*x - sqrt(2346)*sh_11_6*x)
    sh_12_19 = (5/184)*sqrt(874)*sh_11_17*z + (5/276)*sqrt(2185)*sh_11_18*y - 5/276*sqrt(115)*sh_11_19*z - 5/276*sqrt(115)*sh_11_3*x - 5/184*sqrt(874)*sh_11_5*x
    sh_12_20 = (5/276)*sqrt(2185)*sh_11_18*z + (5/69)*sqrt(115)*sh_11_19*y - 5/276*sqrt(69)*sh_11_2*x - 5/276*sqrt(69)*sh_11_20*z - 5/276*sqrt(2185)*sh_11_4*x
    sh_12_21 = -5/552*sqrt(138)*sh_11_1*x + (5/276)*sqrt(2415)*sh_11_19*z + (5/92)*sqrt(161)*sh_11_20*y - 5/552*sqrt(138)*sh_11_21*z - 5/276*sqrt(2415)*sh_11_3*x
    sh_12_22 = -5/552*sqrt(46)*sh_11_0*x - 5/552*sqrt(10626)*sh_11_2*x + (5/552)*sqrt(10626)*sh_11_20*z + (5/138)*sqrt(253)*sh_11_21*y - 5/552*sqrt(46)*sh_11_22*z
    sh_12_23 = -5/24*sqrt(22)*sh_11_1*x + (5/24)*sqrt(22)*sh_11_21*z + (5/12)*sh_11_22*y
    sh_12_24 = (5/12)*sqrt(6)*(-sh_11_0*x + sh_11_22*z)
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12,
        sh_7_0, sh_7_1, sh_7_2, sh_7_3, sh_7_4, sh_7_5, sh_7_6, sh_7_7, sh_7_8, sh_7_9, sh_7_10, sh_7_11, sh_7_12, sh_7_13, sh_7_14,
        sh_8_0, sh_8_1, sh_8_2, sh_8_3, sh_8_4, sh_8_5, sh_8_6, sh_8_7, sh_8_8, sh_8_9, sh_8_10, sh_8_11, sh_8_12, sh_8_13, sh_8_14, sh_8_15, sh_8_16,
        sh_9_0, sh_9_1, sh_9_2, sh_9_3, sh_9_4, sh_9_5, sh_9_6, sh_9_7, sh_9_8, sh_9_9, sh_9_10, sh_9_11, sh_9_12, sh_9_13, sh_9_14, sh_9_15, sh_9_16, sh_9_17, sh_9_18,
        sh_10_0, sh_10_1, sh_10_2, sh_10_3, sh_10_4, sh_10_5, sh_10_6, sh_10_7, sh_10_8, sh_10_9, sh_10_10, sh_10_11, sh_10_12, sh_10_13, sh_10_14, sh_10_15, sh_10_16, sh_10_17, sh_10_18, sh_10_19, sh_10_20,
        sh_11_0, sh_11_1, sh_11_2, sh_11_3, sh_11_4, sh_11_5, sh_11_6, sh_11_7, sh_11_8, sh_11_9, sh_11_10, sh_11_11, sh_11_12, sh_11_13, sh_11_14, sh_11_15, sh_11_16, sh_11_17, sh_11_18, sh_11_19, sh_11_20, sh_11_21, sh_11_22,
        sh_12_0, sh_12_1, sh_12_2, sh_12_3, sh_12_4, sh_12_5, sh_12_6, sh_12_7, sh_12_8, sh_12_9, sh_12_10, sh_12_11, sh_12_12, sh_12_13, sh_12_14, sh_12_15, sh_12_16, sh_12_17, sh_12_18, sh_12_19, sh_12_20, sh_12_21, sh_12_22, sh_12_23, sh_12_24
    ], dim=-1)


_spherical_harmonics = {0: _sph_lmax_0, 1: _sph_lmax_1, 2: _sph_lmax_2, 3: _sph_lmax_3, 4: _sph_lmax_4, 5: _sph_lmax_5, 6: _sph_lmax_6, 7: _sph_lmax_7, 8: _sph_lmax_8, 9: _sph_lmax_9, 10: _sph_lmax_10, 11: _sph_lmax_11, 12: _sph_lmax_12}


@functools.lru_cache
def _get_spherical_harmonics(lmax: int):
    return torch.jit.script(_spherical_harmonics[lmax])
