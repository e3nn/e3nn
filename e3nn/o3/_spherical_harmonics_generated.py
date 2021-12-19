# flake8: noqa
import torch


@torch.jit.script
def _sph_lmax_0_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    return torch.stack([
        sh_0_0
    ], dim=-1)


@torch.jit.script
def _sph_lmax_1_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2
    ], dim=-1)


@torch.jit.script
def _sph_lmax_2_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
    ], dim=-1)


@torch.jit.script
def _sph_lmax_3_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
    ], dim=-1)


@torch.jit.script
def _sph_lmax_4_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
    ], dim=-1)


@torch.jit.script
def _sph_lmax_5_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
    ], dim=-1)


@torch.jit.script
def _sph_lmax_6_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
    ], dim=-1)


@torch.jit.script
def _sph_lmax_7_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    sh_7_0 = sqrt(15) / sqrt(4*pi) * sh_7_0
    sh_7_1 = sqrt(15) / sqrt(4*pi) * sh_7_1
    sh_7_2 = sqrt(15) / sqrt(4*pi) * sh_7_2
    sh_7_3 = sqrt(15) / sqrt(4*pi) * sh_7_3
    sh_7_4 = sqrt(15) / sqrt(4*pi) * sh_7_4
    sh_7_5 = sqrt(15) / sqrt(4*pi) * sh_7_5
    sh_7_6 = sqrt(15) / sqrt(4*pi) * sh_7_6
    sh_7_7 = sqrt(15) / sqrt(4*pi) * sh_7_7
    sh_7_8 = sqrt(15) / sqrt(4*pi) * sh_7_8
    sh_7_9 = sqrt(15) / sqrt(4*pi) * sh_7_9
    sh_7_10 = sqrt(15) / sqrt(4*pi) * sh_7_10
    sh_7_11 = sqrt(15) / sqrt(4*pi) * sh_7_11
    sh_7_12 = sqrt(15) / sqrt(4*pi) * sh_7_12
    sh_7_13 = sqrt(15) / sqrt(4*pi) * sh_7_13
    sh_7_14 = sqrt(15) / sqrt(4*pi) * sh_7_14
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


@torch.jit.script
def _sph_lmax_8_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    sh_7_0 = sqrt(15) / sqrt(4*pi) * sh_7_0
    sh_7_1 = sqrt(15) / sqrt(4*pi) * sh_7_1
    sh_7_2 = sqrt(15) / sqrt(4*pi) * sh_7_2
    sh_7_3 = sqrt(15) / sqrt(4*pi) * sh_7_3
    sh_7_4 = sqrt(15) / sqrt(4*pi) * sh_7_4
    sh_7_5 = sqrt(15) / sqrt(4*pi) * sh_7_5
    sh_7_6 = sqrt(15) / sqrt(4*pi) * sh_7_6
    sh_7_7 = sqrt(15) / sqrt(4*pi) * sh_7_7
    sh_7_8 = sqrt(15) / sqrt(4*pi) * sh_7_8
    sh_7_9 = sqrt(15) / sqrt(4*pi) * sh_7_9
    sh_7_10 = sqrt(15) / sqrt(4*pi) * sh_7_10
    sh_7_11 = sqrt(15) / sqrt(4*pi) * sh_7_11
    sh_7_12 = sqrt(15) / sqrt(4*pi) * sh_7_12
    sh_7_13 = sqrt(15) / sqrt(4*pi) * sh_7_13
    sh_7_14 = sqrt(15) / sqrt(4*pi) * sh_7_14
    sh_8_0 = sqrt(17) / sqrt(4*pi) * sh_8_0
    sh_8_1 = sqrt(17) / sqrt(4*pi) * sh_8_1
    sh_8_2 = sqrt(17) / sqrt(4*pi) * sh_8_2
    sh_8_3 = sqrt(17) / sqrt(4*pi) * sh_8_3
    sh_8_4 = sqrt(17) / sqrt(4*pi) * sh_8_4
    sh_8_5 = sqrt(17) / sqrt(4*pi) * sh_8_5
    sh_8_6 = sqrt(17) / sqrt(4*pi) * sh_8_6
    sh_8_7 = sqrt(17) / sqrt(4*pi) * sh_8_7
    sh_8_8 = sqrt(17) / sqrt(4*pi) * sh_8_8
    sh_8_9 = sqrt(17) / sqrt(4*pi) * sh_8_9
    sh_8_10 = sqrt(17) / sqrt(4*pi) * sh_8_10
    sh_8_11 = sqrt(17) / sqrt(4*pi) * sh_8_11
    sh_8_12 = sqrt(17) / sqrt(4*pi) * sh_8_12
    sh_8_13 = sqrt(17) / sqrt(4*pi) * sh_8_13
    sh_8_14 = sqrt(17) / sqrt(4*pi) * sh_8_14
    sh_8_15 = sqrt(17) / sqrt(4*pi) * sh_8_15
    sh_8_16 = sqrt(17) / sqrt(4*pi) * sh_8_16
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


@torch.jit.script
def _sph_lmax_9_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    sh_7_0 = sqrt(15) / sqrt(4*pi) * sh_7_0
    sh_7_1 = sqrt(15) / sqrt(4*pi) * sh_7_1
    sh_7_2 = sqrt(15) / sqrt(4*pi) * sh_7_2
    sh_7_3 = sqrt(15) / sqrt(4*pi) * sh_7_3
    sh_7_4 = sqrt(15) / sqrt(4*pi) * sh_7_4
    sh_7_5 = sqrt(15) / sqrt(4*pi) * sh_7_5
    sh_7_6 = sqrt(15) / sqrt(4*pi) * sh_7_6
    sh_7_7 = sqrt(15) / sqrt(4*pi) * sh_7_7
    sh_7_8 = sqrt(15) / sqrt(4*pi) * sh_7_8
    sh_7_9 = sqrt(15) / sqrt(4*pi) * sh_7_9
    sh_7_10 = sqrt(15) / sqrt(4*pi) * sh_7_10
    sh_7_11 = sqrt(15) / sqrt(4*pi) * sh_7_11
    sh_7_12 = sqrt(15) / sqrt(4*pi) * sh_7_12
    sh_7_13 = sqrt(15) / sqrt(4*pi) * sh_7_13
    sh_7_14 = sqrt(15) / sqrt(4*pi) * sh_7_14
    sh_8_0 = sqrt(17) / sqrt(4*pi) * sh_8_0
    sh_8_1 = sqrt(17) / sqrt(4*pi) * sh_8_1
    sh_8_2 = sqrt(17) / sqrt(4*pi) * sh_8_2
    sh_8_3 = sqrt(17) / sqrt(4*pi) * sh_8_3
    sh_8_4 = sqrt(17) / sqrt(4*pi) * sh_8_4
    sh_8_5 = sqrt(17) / sqrt(4*pi) * sh_8_5
    sh_8_6 = sqrt(17) / sqrt(4*pi) * sh_8_6
    sh_8_7 = sqrt(17) / sqrt(4*pi) * sh_8_7
    sh_8_8 = sqrt(17) / sqrt(4*pi) * sh_8_8
    sh_8_9 = sqrt(17) / sqrt(4*pi) * sh_8_9
    sh_8_10 = sqrt(17) / sqrt(4*pi) * sh_8_10
    sh_8_11 = sqrt(17) / sqrt(4*pi) * sh_8_11
    sh_8_12 = sqrt(17) / sqrt(4*pi) * sh_8_12
    sh_8_13 = sqrt(17) / sqrt(4*pi) * sh_8_13
    sh_8_14 = sqrt(17) / sqrt(4*pi) * sh_8_14
    sh_8_15 = sqrt(17) / sqrt(4*pi) * sh_8_15
    sh_8_16 = sqrt(17) / sqrt(4*pi) * sh_8_16
    sh_9_0 = sqrt(19) / sqrt(4*pi) * sh_9_0
    sh_9_1 = sqrt(19) / sqrt(4*pi) * sh_9_1
    sh_9_2 = sqrt(19) / sqrt(4*pi) * sh_9_2
    sh_9_3 = sqrt(19) / sqrt(4*pi) * sh_9_3
    sh_9_4 = sqrt(19) / sqrt(4*pi) * sh_9_4
    sh_9_5 = sqrt(19) / sqrt(4*pi) * sh_9_5
    sh_9_6 = sqrt(19) / sqrt(4*pi) * sh_9_6
    sh_9_7 = sqrt(19) / sqrt(4*pi) * sh_9_7
    sh_9_8 = sqrt(19) / sqrt(4*pi) * sh_9_8
    sh_9_9 = sqrt(19) / sqrt(4*pi) * sh_9_9
    sh_9_10 = sqrt(19) / sqrt(4*pi) * sh_9_10
    sh_9_11 = sqrt(19) / sqrt(4*pi) * sh_9_11
    sh_9_12 = sqrt(19) / sqrt(4*pi) * sh_9_12
    sh_9_13 = sqrt(19) / sqrt(4*pi) * sh_9_13
    sh_9_14 = sqrt(19) / sqrt(4*pi) * sh_9_14
    sh_9_15 = sqrt(19) / sqrt(4*pi) * sh_9_15
    sh_9_16 = sqrt(19) / sqrt(4*pi) * sh_9_16
    sh_9_17 = sqrt(19) / sqrt(4*pi) * sh_9_17
    sh_9_18 = sqrt(19) / sqrt(4*pi) * sh_9_18
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


@torch.jit.script
def _sph_lmax_10_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    sh_7_0 = sqrt(15) / sqrt(4*pi) * sh_7_0
    sh_7_1 = sqrt(15) / sqrt(4*pi) * sh_7_1
    sh_7_2 = sqrt(15) / sqrt(4*pi) * sh_7_2
    sh_7_3 = sqrt(15) / sqrt(4*pi) * sh_7_3
    sh_7_4 = sqrt(15) / sqrt(4*pi) * sh_7_4
    sh_7_5 = sqrt(15) / sqrt(4*pi) * sh_7_5
    sh_7_6 = sqrt(15) / sqrt(4*pi) * sh_7_6
    sh_7_7 = sqrt(15) / sqrt(4*pi) * sh_7_7
    sh_7_8 = sqrt(15) / sqrt(4*pi) * sh_7_8
    sh_7_9 = sqrt(15) / sqrt(4*pi) * sh_7_9
    sh_7_10 = sqrt(15) / sqrt(4*pi) * sh_7_10
    sh_7_11 = sqrt(15) / sqrt(4*pi) * sh_7_11
    sh_7_12 = sqrt(15) / sqrt(4*pi) * sh_7_12
    sh_7_13 = sqrt(15) / sqrt(4*pi) * sh_7_13
    sh_7_14 = sqrt(15) / sqrt(4*pi) * sh_7_14
    sh_8_0 = sqrt(17) / sqrt(4*pi) * sh_8_0
    sh_8_1 = sqrt(17) / sqrt(4*pi) * sh_8_1
    sh_8_2 = sqrt(17) / sqrt(4*pi) * sh_8_2
    sh_8_3 = sqrt(17) / sqrt(4*pi) * sh_8_3
    sh_8_4 = sqrt(17) / sqrt(4*pi) * sh_8_4
    sh_8_5 = sqrt(17) / sqrt(4*pi) * sh_8_5
    sh_8_6 = sqrt(17) / sqrt(4*pi) * sh_8_6
    sh_8_7 = sqrt(17) / sqrt(4*pi) * sh_8_7
    sh_8_8 = sqrt(17) / sqrt(4*pi) * sh_8_8
    sh_8_9 = sqrt(17) / sqrt(4*pi) * sh_8_9
    sh_8_10 = sqrt(17) / sqrt(4*pi) * sh_8_10
    sh_8_11 = sqrt(17) / sqrt(4*pi) * sh_8_11
    sh_8_12 = sqrt(17) / sqrt(4*pi) * sh_8_12
    sh_8_13 = sqrt(17) / sqrt(4*pi) * sh_8_13
    sh_8_14 = sqrt(17) / sqrt(4*pi) * sh_8_14
    sh_8_15 = sqrt(17) / sqrt(4*pi) * sh_8_15
    sh_8_16 = sqrt(17) / sqrt(4*pi) * sh_8_16
    sh_9_0 = sqrt(19) / sqrt(4*pi) * sh_9_0
    sh_9_1 = sqrt(19) / sqrt(4*pi) * sh_9_1
    sh_9_2 = sqrt(19) / sqrt(4*pi) * sh_9_2
    sh_9_3 = sqrt(19) / sqrt(4*pi) * sh_9_3
    sh_9_4 = sqrt(19) / sqrt(4*pi) * sh_9_4
    sh_9_5 = sqrt(19) / sqrt(4*pi) * sh_9_5
    sh_9_6 = sqrt(19) / sqrt(4*pi) * sh_9_6
    sh_9_7 = sqrt(19) / sqrt(4*pi) * sh_9_7
    sh_9_8 = sqrt(19) / sqrt(4*pi) * sh_9_8
    sh_9_9 = sqrt(19) / sqrt(4*pi) * sh_9_9
    sh_9_10 = sqrt(19) / sqrt(4*pi) * sh_9_10
    sh_9_11 = sqrt(19) / sqrt(4*pi) * sh_9_11
    sh_9_12 = sqrt(19) / sqrt(4*pi) * sh_9_12
    sh_9_13 = sqrt(19) / sqrt(4*pi) * sh_9_13
    sh_9_14 = sqrt(19) / sqrt(4*pi) * sh_9_14
    sh_9_15 = sqrt(19) / sqrt(4*pi) * sh_9_15
    sh_9_16 = sqrt(19) / sqrt(4*pi) * sh_9_16
    sh_9_17 = sqrt(19) / sqrt(4*pi) * sh_9_17
    sh_9_18 = sqrt(19) / sqrt(4*pi) * sh_9_18
    sh_10_0 = sqrt(21) / sqrt(4*pi) * sh_10_0
    sh_10_1 = sqrt(21) / sqrt(4*pi) * sh_10_1
    sh_10_2 = sqrt(21) / sqrt(4*pi) * sh_10_2
    sh_10_3 = sqrt(21) / sqrt(4*pi) * sh_10_3
    sh_10_4 = sqrt(21) / sqrt(4*pi) * sh_10_4
    sh_10_5 = sqrt(21) / sqrt(4*pi) * sh_10_5
    sh_10_6 = sqrt(21) / sqrt(4*pi) * sh_10_6
    sh_10_7 = sqrt(21) / sqrt(4*pi) * sh_10_7
    sh_10_8 = sqrt(21) / sqrt(4*pi) * sh_10_8
    sh_10_9 = sqrt(21) / sqrt(4*pi) * sh_10_9
    sh_10_10 = sqrt(21) / sqrt(4*pi) * sh_10_10
    sh_10_11 = sqrt(21) / sqrt(4*pi) * sh_10_11
    sh_10_12 = sqrt(21) / sqrt(4*pi) * sh_10_12
    sh_10_13 = sqrt(21) / sqrt(4*pi) * sh_10_13
    sh_10_14 = sqrt(21) / sqrt(4*pi) * sh_10_14
    sh_10_15 = sqrt(21) / sqrt(4*pi) * sh_10_15
    sh_10_16 = sqrt(21) / sqrt(4*pi) * sh_10_16
    sh_10_17 = sqrt(21) / sqrt(4*pi) * sh_10_17
    sh_10_18 = sqrt(21) / sqrt(4*pi) * sh_10_18
    sh_10_19 = sqrt(21) / sqrt(4*pi) * sh_10_19
    sh_10_20 = sqrt(21) / sqrt(4*pi) * sh_10_20
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


@torch.jit.script
def _sph_lmax_11_integral(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
    sh_11_0 = sqrt(462)*(sh_10_0*z + sh_10_20*x)/22
    sh_11_1 = sqrt(21)*sh_10_0*y/11 + sqrt(105)*sh_10_1*z/11 + sqrt(105)*sh_10_19*x/11
    sh_11_2 = -sqrt(2)*sh_10_0*z/22 + 2*sqrt(10)*sh_10_1*y/11 + sqrt(95)*sh_10_18*x/11 + sqrt(95)*sh_10_2*z/11 + sqrt(2)*sh_10_20*x/22
    sh_11_3 = -sqrt(6)*sh_10_1*z/22 + 3*sqrt(38)*sh_10_17*x/22 + sqrt(6)*sh_10_19*x/22 + sqrt(57)*sh_10_2*y/11 + 3*sqrt(38)*sh_10_3*z/22
    sh_11_4 = 3*sqrt(34)*sh_10_16*x/22 + sqrt(3)*sh_10_18*x/11 - sqrt(3)*sh_10_2*z/11 + 6*sqrt(2)*sh_10_3*y/11 + 3*sqrt(34)*sh_10_4*z/22
    sh_11_5 = 2*sqrt(17)*sh_10_15*x/11 + sqrt(5)*sh_10_17*x/11 - sqrt(5)*sh_10_3*z/11 + sqrt(85)*sh_10_4*y/11 + 2*sqrt(17)*sh_10_5*z/11
    sh_11_6 = 2*sqrt(15)*sh_10_14*x/11 + sqrt(30)*sh_10_16*x/22 - sqrt(30)*sh_10_4*z/22 + 4*sqrt(6)*sh_10_5*y/11 + 2*sqrt(15)*sh_10_6*z/11
    sh_11_7 = sqrt(210)*sh_10_13*x/22 + sqrt(42)*sh_10_15*x/22 - sqrt(42)*sh_10_5*z/22 + sqrt(105)*sh_10_6*y/11 + sqrt(210)*sh_10_7*z/22
    sh_11_8 = sqrt(182)*sh_10_12*x/22 + sqrt(14)*sh_10_14*x/11 - sqrt(14)*sh_10_6*z/11 + 4*sqrt(7)*sh_10_7*y/11 + sqrt(182)*sh_10_8*z/22
    sh_11_9 = sqrt(5313)*(sqrt(23023)*sh_10_11*x + sqrt(10626)*sh_10_13*x - sqrt(10626)*sh_10_7*z + sqrt(69069)*sh_10_8*y + sqrt(23023)*sh_10_9*z)/19481
    sh_11_10 = sqrt(66)*sh_10_10*x/11 + 3*sqrt(10)*sh_10_12*x/22 - 3*sqrt(10)*sh_10_8*z/22 + 2*sqrt(30)*sh_10_9*y/11
    sh_11_11 = sh_10_10*y - sqrt(55)*sh_10_11*z/11 - sqrt(55)*sh_10_9*x/11
    sh_11_12 = sqrt(66)*sh_10_10*z/11 + 2*sqrt(30)*sh_10_11*y/11 - 3*sqrt(10)*sh_10_12*z/22 - 3*sqrt(10)*sh_10_8*x/22
    sh_11_13 = sqrt(5313)*(sqrt(23023)*sh_10_11*z + sqrt(69069)*sh_10_12*y - sqrt(10626)*sh_10_13*z - sqrt(10626)*sh_10_7*x - sqrt(23023)*sh_10_9*x)/19481
    sh_11_14 = sqrt(182)*sh_10_12*z/22 + 4*sqrt(7)*sh_10_13*y/11 - sqrt(14)*sh_10_14*z/11 - sqrt(14)*sh_10_6*x/11 - sqrt(182)*sh_10_8*x/22
    sh_11_15 = sqrt(210)*sh_10_13*z/22 + sqrt(105)*sh_10_14*y/11 - sqrt(42)*sh_10_15*z/22 - sqrt(42)*sh_10_5*x/22 - sqrt(210)*sh_10_7*x/22
    sh_11_16 = 2*sqrt(15)*sh_10_14*z/11 + 4*sqrt(6)*sh_10_15*y/11 - sqrt(30)*sh_10_16*z/22 - sqrt(30)*sh_10_4*x/22 - 2*sqrt(15)*sh_10_6*x/11
    sh_11_17 = 2*sqrt(17)*sh_10_15*z/11 + sqrt(85)*sh_10_16*y/11 - sqrt(5)*sh_10_17*z/11 - sqrt(5)*sh_10_3*x/11 - 2*sqrt(17)*sh_10_5*x/11
    sh_11_18 = 3*sqrt(34)*sh_10_16*z/22 + 6*sqrt(2)*sh_10_17*y/11 - sqrt(3)*sh_10_18*z/11 - sqrt(3)*sh_10_2*x/11 - 3*sqrt(34)*sh_10_4*x/22
    sh_11_19 = -sqrt(6)*sh_10_1*x/22 + 3*sqrt(38)*sh_10_17*z/22 + sqrt(57)*sh_10_18*y/11 - sqrt(6)*sh_10_19*z/22 - 3*sqrt(38)*sh_10_3*x/22
    sh_11_20 = -sqrt(2)*sh_10_0*x/22 + sqrt(95)*sh_10_18*z/11 + 2*sqrt(10)*sh_10_19*y/11 - sqrt(95)*sh_10_2*x/11 - sqrt(2)*sh_10_20*z/22
    sh_11_21 = -sqrt(105)*sh_10_1*x/11 + sqrt(105)*sh_10_19*z/11 + sqrt(21)*sh_10_20*y/11
    sh_11_22 = sqrt(462)*(-sh_10_0*x + sh_10_20*z)/22
    sh_0_0 = sqrt(1) / sqrt(4*pi) * sh_0_0
    sh_1_0 = sqrt(3) / sqrt(4*pi) * sh_1_0
    sh_1_1 = sqrt(3) / sqrt(4*pi) * sh_1_1
    sh_1_2 = sqrt(3) / sqrt(4*pi) * sh_1_2
    sh_2_0 = sqrt(5) / sqrt(4*pi) * sh_2_0
    sh_2_1 = sqrt(5) / sqrt(4*pi) * sh_2_1
    sh_2_2 = sqrt(5) / sqrt(4*pi) * sh_2_2
    sh_2_3 = sqrt(5) / sqrt(4*pi) * sh_2_3
    sh_2_4 = sqrt(5) / sqrt(4*pi) * sh_2_4
    sh_3_0 = sqrt(7) / sqrt(4*pi) * sh_3_0
    sh_3_1 = sqrt(7) / sqrt(4*pi) * sh_3_1
    sh_3_2 = sqrt(7) / sqrt(4*pi) * sh_3_2
    sh_3_3 = sqrt(7) / sqrt(4*pi) * sh_3_3
    sh_3_4 = sqrt(7) / sqrt(4*pi) * sh_3_4
    sh_3_5 = sqrt(7) / sqrt(4*pi) * sh_3_5
    sh_3_6 = sqrt(7) / sqrt(4*pi) * sh_3_6
    sh_4_0 = sqrt(9) / sqrt(4*pi) * sh_4_0
    sh_4_1 = sqrt(9) / sqrt(4*pi) * sh_4_1
    sh_4_2 = sqrt(9) / sqrt(4*pi) * sh_4_2
    sh_4_3 = sqrt(9) / sqrt(4*pi) * sh_4_3
    sh_4_4 = sqrt(9) / sqrt(4*pi) * sh_4_4
    sh_4_5 = sqrt(9) / sqrt(4*pi) * sh_4_5
    sh_4_6 = sqrt(9) / sqrt(4*pi) * sh_4_6
    sh_4_7 = sqrt(9) / sqrt(4*pi) * sh_4_7
    sh_4_8 = sqrt(9) / sqrt(4*pi) * sh_4_8
    sh_5_0 = sqrt(11) / sqrt(4*pi) * sh_5_0
    sh_5_1 = sqrt(11) / sqrt(4*pi) * sh_5_1
    sh_5_2 = sqrt(11) / sqrt(4*pi) * sh_5_2
    sh_5_3 = sqrt(11) / sqrt(4*pi) * sh_5_3
    sh_5_4 = sqrt(11) / sqrt(4*pi) * sh_5_4
    sh_5_5 = sqrt(11) / sqrt(4*pi) * sh_5_5
    sh_5_6 = sqrt(11) / sqrt(4*pi) * sh_5_6
    sh_5_7 = sqrt(11) / sqrt(4*pi) * sh_5_7
    sh_5_8 = sqrt(11) / sqrt(4*pi) * sh_5_8
    sh_5_9 = sqrt(11) / sqrt(4*pi) * sh_5_9
    sh_5_10 = sqrt(11) / sqrt(4*pi) * sh_5_10
    sh_6_0 = sqrt(13) / sqrt(4*pi) * sh_6_0
    sh_6_1 = sqrt(13) / sqrt(4*pi) * sh_6_1
    sh_6_2 = sqrt(13) / sqrt(4*pi) * sh_6_2
    sh_6_3 = sqrt(13) / sqrt(4*pi) * sh_6_3
    sh_6_4 = sqrt(13) / sqrt(4*pi) * sh_6_4
    sh_6_5 = sqrt(13) / sqrt(4*pi) * sh_6_5
    sh_6_6 = sqrt(13) / sqrt(4*pi) * sh_6_6
    sh_6_7 = sqrt(13) / sqrt(4*pi) * sh_6_7
    sh_6_8 = sqrt(13) / sqrt(4*pi) * sh_6_8
    sh_6_9 = sqrt(13) / sqrt(4*pi) * sh_6_9
    sh_6_10 = sqrt(13) / sqrt(4*pi) * sh_6_10
    sh_6_11 = sqrt(13) / sqrt(4*pi) * sh_6_11
    sh_6_12 = sqrt(13) / sqrt(4*pi) * sh_6_12
    sh_7_0 = sqrt(15) / sqrt(4*pi) * sh_7_0
    sh_7_1 = sqrt(15) / sqrt(4*pi) * sh_7_1
    sh_7_2 = sqrt(15) / sqrt(4*pi) * sh_7_2
    sh_7_3 = sqrt(15) / sqrt(4*pi) * sh_7_3
    sh_7_4 = sqrt(15) / sqrt(4*pi) * sh_7_4
    sh_7_5 = sqrt(15) / sqrt(4*pi) * sh_7_5
    sh_7_6 = sqrt(15) / sqrt(4*pi) * sh_7_6
    sh_7_7 = sqrt(15) / sqrt(4*pi) * sh_7_7
    sh_7_8 = sqrt(15) / sqrt(4*pi) * sh_7_8
    sh_7_9 = sqrt(15) / sqrt(4*pi) * sh_7_9
    sh_7_10 = sqrt(15) / sqrt(4*pi) * sh_7_10
    sh_7_11 = sqrt(15) / sqrt(4*pi) * sh_7_11
    sh_7_12 = sqrt(15) / sqrt(4*pi) * sh_7_12
    sh_7_13 = sqrt(15) / sqrt(4*pi) * sh_7_13
    sh_7_14 = sqrt(15) / sqrt(4*pi) * sh_7_14
    sh_8_0 = sqrt(17) / sqrt(4*pi) * sh_8_0
    sh_8_1 = sqrt(17) / sqrt(4*pi) * sh_8_1
    sh_8_2 = sqrt(17) / sqrt(4*pi) * sh_8_2
    sh_8_3 = sqrt(17) / sqrt(4*pi) * sh_8_3
    sh_8_4 = sqrt(17) / sqrt(4*pi) * sh_8_4
    sh_8_5 = sqrt(17) / sqrt(4*pi) * sh_8_5
    sh_8_6 = sqrt(17) / sqrt(4*pi) * sh_8_6
    sh_8_7 = sqrt(17) / sqrt(4*pi) * sh_8_7
    sh_8_8 = sqrt(17) / sqrt(4*pi) * sh_8_8
    sh_8_9 = sqrt(17) / sqrt(4*pi) * sh_8_9
    sh_8_10 = sqrt(17) / sqrt(4*pi) * sh_8_10
    sh_8_11 = sqrt(17) / sqrt(4*pi) * sh_8_11
    sh_8_12 = sqrt(17) / sqrt(4*pi) * sh_8_12
    sh_8_13 = sqrt(17) / sqrt(4*pi) * sh_8_13
    sh_8_14 = sqrt(17) / sqrt(4*pi) * sh_8_14
    sh_8_15 = sqrt(17) / sqrt(4*pi) * sh_8_15
    sh_8_16 = sqrt(17) / sqrt(4*pi) * sh_8_16
    sh_9_0 = sqrt(19) / sqrt(4*pi) * sh_9_0
    sh_9_1 = sqrt(19) / sqrt(4*pi) * sh_9_1
    sh_9_2 = sqrt(19) / sqrt(4*pi) * sh_9_2
    sh_9_3 = sqrt(19) / sqrt(4*pi) * sh_9_3
    sh_9_4 = sqrt(19) / sqrt(4*pi) * sh_9_4
    sh_9_5 = sqrt(19) / sqrt(4*pi) * sh_9_5
    sh_9_6 = sqrt(19) / sqrt(4*pi) * sh_9_6
    sh_9_7 = sqrt(19) / sqrt(4*pi) * sh_9_7
    sh_9_8 = sqrt(19) / sqrt(4*pi) * sh_9_8
    sh_9_9 = sqrt(19) / sqrt(4*pi) * sh_9_9
    sh_9_10 = sqrt(19) / sqrt(4*pi) * sh_9_10
    sh_9_11 = sqrt(19) / sqrt(4*pi) * sh_9_11
    sh_9_12 = sqrt(19) / sqrt(4*pi) * sh_9_12
    sh_9_13 = sqrt(19) / sqrt(4*pi) * sh_9_13
    sh_9_14 = sqrt(19) / sqrt(4*pi) * sh_9_14
    sh_9_15 = sqrt(19) / sqrt(4*pi) * sh_9_15
    sh_9_16 = sqrt(19) / sqrt(4*pi) * sh_9_16
    sh_9_17 = sqrt(19) / sqrt(4*pi) * sh_9_17
    sh_9_18 = sqrt(19) / sqrt(4*pi) * sh_9_18
    sh_10_0 = sqrt(21) / sqrt(4*pi) * sh_10_0
    sh_10_1 = sqrt(21) / sqrt(4*pi) * sh_10_1
    sh_10_2 = sqrt(21) / sqrt(4*pi) * sh_10_2
    sh_10_3 = sqrt(21) / sqrt(4*pi) * sh_10_3
    sh_10_4 = sqrt(21) / sqrt(4*pi) * sh_10_4
    sh_10_5 = sqrt(21) / sqrt(4*pi) * sh_10_5
    sh_10_6 = sqrt(21) / sqrt(4*pi) * sh_10_6
    sh_10_7 = sqrt(21) / sqrt(4*pi) * sh_10_7
    sh_10_8 = sqrt(21) / sqrt(4*pi) * sh_10_8
    sh_10_9 = sqrt(21) / sqrt(4*pi) * sh_10_9
    sh_10_10 = sqrt(21) / sqrt(4*pi) * sh_10_10
    sh_10_11 = sqrt(21) / sqrt(4*pi) * sh_10_11
    sh_10_12 = sqrt(21) / sqrt(4*pi) * sh_10_12
    sh_10_13 = sqrt(21) / sqrt(4*pi) * sh_10_13
    sh_10_14 = sqrt(21) / sqrt(4*pi) * sh_10_14
    sh_10_15 = sqrt(21) / sqrt(4*pi) * sh_10_15
    sh_10_16 = sqrt(21) / sqrt(4*pi) * sh_10_16
    sh_10_17 = sqrt(21) / sqrt(4*pi) * sh_10_17
    sh_10_18 = sqrt(21) / sqrt(4*pi) * sh_10_18
    sh_10_19 = sqrt(21) / sqrt(4*pi) * sh_10_19
    sh_10_20 = sqrt(21) / sqrt(4*pi) * sh_10_20
    sh_11_0 = sqrt(23) / sqrt(4*pi) * sh_11_0
    sh_11_1 = sqrt(23) / sqrt(4*pi) * sh_11_1
    sh_11_2 = sqrt(23) / sqrt(4*pi) * sh_11_2
    sh_11_3 = sqrt(23) / sqrt(4*pi) * sh_11_3
    sh_11_4 = sqrt(23) / sqrt(4*pi) * sh_11_4
    sh_11_5 = sqrt(23) / sqrt(4*pi) * sh_11_5
    sh_11_6 = sqrt(23) / sqrt(4*pi) * sh_11_6
    sh_11_7 = sqrt(23) / sqrt(4*pi) * sh_11_7
    sh_11_8 = sqrt(23) / sqrt(4*pi) * sh_11_8
    sh_11_9 = sqrt(23) / sqrt(4*pi) * sh_11_9
    sh_11_10 = sqrt(23) / sqrt(4*pi) * sh_11_10
    sh_11_11 = sqrt(23) / sqrt(4*pi) * sh_11_11
    sh_11_12 = sqrt(23) / sqrt(4*pi) * sh_11_12
    sh_11_13 = sqrt(23) / sqrt(4*pi) * sh_11_13
    sh_11_14 = sqrt(23) / sqrt(4*pi) * sh_11_14
    sh_11_15 = sqrt(23) / sqrt(4*pi) * sh_11_15
    sh_11_16 = sqrt(23) / sqrt(4*pi) * sh_11_16
    sh_11_17 = sqrt(23) / sqrt(4*pi) * sh_11_17
    sh_11_18 = sqrt(23) / sqrt(4*pi) * sh_11_18
    sh_11_19 = sqrt(23) / sqrt(4*pi) * sh_11_19
    sh_11_20 = sqrt(23) / sqrt(4*pi) * sh_11_20
    sh_11_21 = sqrt(23) / sqrt(4*pi) * sh_11_21
    sh_11_22 = sqrt(23) / sqrt(4*pi) * sh_11_22
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


@torch.jit.script
def _sph_lmax_0_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    return torch.stack([
        sh_0_0
    ], dim=-1)


@torch.jit.script
def _sph_lmax_1_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2
    ], dim=-1)


@torch.jit.script
def _sph_lmax_2_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
    ], dim=-1)


@torch.jit.script
def _sph_lmax_3_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
    ], dim=-1)


@torch.jit.script
def _sph_lmax_4_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
    ], dim=-1)


@torch.jit.script
def _sph_lmax_5_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
    ], dim=-1)


@torch.jit.script
def _sph_lmax_6_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
    ], dim=-1)


@torch.jit.script
def _sph_lmax_7_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    sh_7_0 = sqrt(15) * sh_7_0
    sh_7_1 = sqrt(15) * sh_7_1
    sh_7_2 = sqrt(15) * sh_7_2
    sh_7_3 = sqrt(15) * sh_7_3
    sh_7_4 = sqrt(15) * sh_7_4
    sh_7_5 = sqrt(15) * sh_7_5
    sh_7_6 = sqrt(15) * sh_7_6
    sh_7_7 = sqrt(15) * sh_7_7
    sh_7_8 = sqrt(15) * sh_7_8
    sh_7_9 = sqrt(15) * sh_7_9
    sh_7_10 = sqrt(15) * sh_7_10
    sh_7_11 = sqrt(15) * sh_7_11
    sh_7_12 = sqrt(15) * sh_7_12
    sh_7_13 = sqrt(15) * sh_7_13
    sh_7_14 = sqrt(15) * sh_7_14
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


@torch.jit.script
def _sph_lmax_8_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    sh_7_0 = sqrt(15) * sh_7_0
    sh_7_1 = sqrt(15) * sh_7_1
    sh_7_2 = sqrt(15) * sh_7_2
    sh_7_3 = sqrt(15) * sh_7_3
    sh_7_4 = sqrt(15) * sh_7_4
    sh_7_5 = sqrt(15) * sh_7_5
    sh_7_6 = sqrt(15) * sh_7_6
    sh_7_7 = sqrt(15) * sh_7_7
    sh_7_8 = sqrt(15) * sh_7_8
    sh_7_9 = sqrt(15) * sh_7_9
    sh_7_10 = sqrt(15) * sh_7_10
    sh_7_11 = sqrt(15) * sh_7_11
    sh_7_12 = sqrt(15) * sh_7_12
    sh_7_13 = sqrt(15) * sh_7_13
    sh_7_14 = sqrt(15) * sh_7_14
    sh_8_0 = sqrt(17) * sh_8_0
    sh_8_1 = sqrt(17) * sh_8_1
    sh_8_2 = sqrt(17) * sh_8_2
    sh_8_3 = sqrt(17) * sh_8_3
    sh_8_4 = sqrt(17) * sh_8_4
    sh_8_5 = sqrt(17) * sh_8_5
    sh_8_6 = sqrt(17) * sh_8_6
    sh_8_7 = sqrt(17) * sh_8_7
    sh_8_8 = sqrt(17) * sh_8_8
    sh_8_9 = sqrt(17) * sh_8_9
    sh_8_10 = sqrt(17) * sh_8_10
    sh_8_11 = sqrt(17) * sh_8_11
    sh_8_12 = sqrt(17) * sh_8_12
    sh_8_13 = sqrt(17) * sh_8_13
    sh_8_14 = sqrt(17) * sh_8_14
    sh_8_15 = sqrt(17) * sh_8_15
    sh_8_16 = sqrt(17) * sh_8_16
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


@torch.jit.script
def _sph_lmax_9_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    sh_7_0 = sqrt(15) * sh_7_0
    sh_7_1 = sqrt(15) * sh_7_1
    sh_7_2 = sqrt(15) * sh_7_2
    sh_7_3 = sqrt(15) * sh_7_3
    sh_7_4 = sqrt(15) * sh_7_4
    sh_7_5 = sqrt(15) * sh_7_5
    sh_7_6 = sqrt(15) * sh_7_6
    sh_7_7 = sqrt(15) * sh_7_7
    sh_7_8 = sqrt(15) * sh_7_8
    sh_7_9 = sqrt(15) * sh_7_9
    sh_7_10 = sqrt(15) * sh_7_10
    sh_7_11 = sqrt(15) * sh_7_11
    sh_7_12 = sqrt(15) * sh_7_12
    sh_7_13 = sqrt(15) * sh_7_13
    sh_7_14 = sqrt(15) * sh_7_14
    sh_8_0 = sqrt(17) * sh_8_0
    sh_8_1 = sqrt(17) * sh_8_1
    sh_8_2 = sqrt(17) * sh_8_2
    sh_8_3 = sqrt(17) * sh_8_3
    sh_8_4 = sqrt(17) * sh_8_4
    sh_8_5 = sqrt(17) * sh_8_5
    sh_8_6 = sqrt(17) * sh_8_6
    sh_8_7 = sqrt(17) * sh_8_7
    sh_8_8 = sqrt(17) * sh_8_8
    sh_8_9 = sqrt(17) * sh_8_9
    sh_8_10 = sqrt(17) * sh_8_10
    sh_8_11 = sqrt(17) * sh_8_11
    sh_8_12 = sqrt(17) * sh_8_12
    sh_8_13 = sqrt(17) * sh_8_13
    sh_8_14 = sqrt(17) * sh_8_14
    sh_8_15 = sqrt(17) * sh_8_15
    sh_8_16 = sqrt(17) * sh_8_16
    sh_9_0 = sqrt(19) * sh_9_0
    sh_9_1 = sqrt(19) * sh_9_1
    sh_9_2 = sqrt(19) * sh_9_2
    sh_9_3 = sqrt(19) * sh_9_3
    sh_9_4 = sqrt(19) * sh_9_4
    sh_9_5 = sqrt(19) * sh_9_5
    sh_9_6 = sqrt(19) * sh_9_6
    sh_9_7 = sqrt(19) * sh_9_7
    sh_9_8 = sqrt(19) * sh_9_8
    sh_9_9 = sqrt(19) * sh_9_9
    sh_9_10 = sqrt(19) * sh_9_10
    sh_9_11 = sqrt(19) * sh_9_11
    sh_9_12 = sqrt(19) * sh_9_12
    sh_9_13 = sqrt(19) * sh_9_13
    sh_9_14 = sqrt(19) * sh_9_14
    sh_9_15 = sqrt(19) * sh_9_15
    sh_9_16 = sqrt(19) * sh_9_16
    sh_9_17 = sqrt(19) * sh_9_17
    sh_9_18 = sqrt(19) * sh_9_18
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


@torch.jit.script
def _sph_lmax_10_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    sh_7_0 = sqrt(15) * sh_7_0
    sh_7_1 = sqrt(15) * sh_7_1
    sh_7_2 = sqrt(15) * sh_7_2
    sh_7_3 = sqrt(15) * sh_7_3
    sh_7_4 = sqrt(15) * sh_7_4
    sh_7_5 = sqrt(15) * sh_7_5
    sh_7_6 = sqrt(15) * sh_7_6
    sh_7_7 = sqrt(15) * sh_7_7
    sh_7_8 = sqrt(15) * sh_7_8
    sh_7_9 = sqrt(15) * sh_7_9
    sh_7_10 = sqrt(15) * sh_7_10
    sh_7_11 = sqrt(15) * sh_7_11
    sh_7_12 = sqrt(15) * sh_7_12
    sh_7_13 = sqrt(15) * sh_7_13
    sh_7_14 = sqrt(15) * sh_7_14
    sh_8_0 = sqrt(17) * sh_8_0
    sh_8_1 = sqrt(17) * sh_8_1
    sh_8_2 = sqrt(17) * sh_8_2
    sh_8_3 = sqrt(17) * sh_8_3
    sh_8_4 = sqrt(17) * sh_8_4
    sh_8_5 = sqrt(17) * sh_8_5
    sh_8_6 = sqrt(17) * sh_8_6
    sh_8_7 = sqrt(17) * sh_8_7
    sh_8_8 = sqrt(17) * sh_8_8
    sh_8_9 = sqrt(17) * sh_8_9
    sh_8_10 = sqrt(17) * sh_8_10
    sh_8_11 = sqrt(17) * sh_8_11
    sh_8_12 = sqrt(17) * sh_8_12
    sh_8_13 = sqrt(17) * sh_8_13
    sh_8_14 = sqrt(17) * sh_8_14
    sh_8_15 = sqrt(17) * sh_8_15
    sh_8_16 = sqrt(17) * sh_8_16
    sh_9_0 = sqrt(19) * sh_9_0
    sh_9_1 = sqrt(19) * sh_9_1
    sh_9_2 = sqrt(19) * sh_9_2
    sh_9_3 = sqrt(19) * sh_9_3
    sh_9_4 = sqrt(19) * sh_9_4
    sh_9_5 = sqrt(19) * sh_9_5
    sh_9_6 = sqrt(19) * sh_9_6
    sh_9_7 = sqrt(19) * sh_9_7
    sh_9_8 = sqrt(19) * sh_9_8
    sh_9_9 = sqrt(19) * sh_9_9
    sh_9_10 = sqrt(19) * sh_9_10
    sh_9_11 = sqrt(19) * sh_9_11
    sh_9_12 = sqrt(19) * sh_9_12
    sh_9_13 = sqrt(19) * sh_9_13
    sh_9_14 = sqrt(19) * sh_9_14
    sh_9_15 = sqrt(19) * sh_9_15
    sh_9_16 = sqrt(19) * sh_9_16
    sh_9_17 = sqrt(19) * sh_9_17
    sh_9_18 = sqrt(19) * sh_9_18
    sh_10_0 = sqrt(21) * sh_10_0
    sh_10_1 = sqrt(21) * sh_10_1
    sh_10_2 = sqrt(21) * sh_10_2
    sh_10_3 = sqrt(21) * sh_10_3
    sh_10_4 = sqrt(21) * sh_10_4
    sh_10_5 = sqrt(21) * sh_10_5
    sh_10_6 = sqrt(21) * sh_10_6
    sh_10_7 = sqrt(21) * sh_10_7
    sh_10_8 = sqrt(21) * sh_10_8
    sh_10_9 = sqrt(21) * sh_10_9
    sh_10_10 = sqrt(21) * sh_10_10
    sh_10_11 = sqrt(21) * sh_10_11
    sh_10_12 = sqrt(21) * sh_10_12
    sh_10_13 = sqrt(21) * sh_10_13
    sh_10_14 = sqrt(21) * sh_10_14
    sh_10_15 = sqrt(21) * sh_10_15
    sh_10_16 = sqrt(21) * sh_10_16
    sh_10_17 = sqrt(21) * sh_10_17
    sh_10_18 = sqrt(21) * sh_10_18
    sh_10_19 = sqrt(21) * sh_10_19
    sh_10_20 = sqrt(21) * sh_10_20
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


@torch.jit.script
def _sph_lmax_11_component(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
    sh_11_0 = sqrt(462)*(sh_10_0*z + sh_10_20*x)/22
    sh_11_1 = sqrt(21)*sh_10_0*y/11 + sqrt(105)*sh_10_1*z/11 + sqrt(105)*sh_10_19*x/11
    sh_11_2 = -sqrt(2)*sh_10_0*z/22 + 2*sqrt(10)*sh_10_1*y/11 + sqrt(95)*sh_10_18*x/11 + sqrt(95)*sh_10_2*z/11 + sqrt(2)*sh_10_20*x/22
    sh_11_3 = -sqrt(6)*sh_10_1*z/22 + 3*sqrt(38)*sh_10_17*x/22 + sqrt(6)*sh_10_19*x/22 + sqrt(57)*sh_10_2*y/11 + 3*sqrt(38)*sh_10_3*z/22
    sh_11_4 = 3*sqrt(34)*sh_10_16*x/22 + sqrt(3)*sh_10_18*x/11 - sqrt(3)*sh_10_2*z/11 + 6*sqrt(2)*sh_10_3*y/11 + 3*sqrt(34)*sh_10_4*z/22
    sh_11_5 = 2*sqrt(17)*sh_10_15*x/11 + sqrt(5)*sh_10_17*x/11 - sqrt(5)*sh_10_3*z/11 + sqrt(85)*sh_10_4*y/11 + 2*sqrt(17)*sh_10_5*z/11
    sh_11_6 = 2*sqrt(15)*sh_10_14*x/11 + sqrt(30)*sh_10_16*x/22 - sqrt(30)*sh_10_4*z/22 + 4*sqrt(6)*sh_10_5*y/11 + 2*sqrt(15)*sh_10_6*z/11
    sh_11_7 = sqrt(210)*sh_10_13*x/22 + sqrt(42)*sh_10_15*x/22 - sqrt(42)*sh_10_5*z/22 + sqrt(105)*sh_10_6*y/11 + sqrt(210)*sh_10_7*z/22
    sh_11_8 = sqrt(182)*sh_10_12*x/22 + sqrt(14)*sh_10_14*x/11 - sqrt(14)*sh_10_6*z/11 + 4*sqrt(7)*sh_10_7*y/11 + sqrt(182)*sh_10_8*z/22
    sh_11_9 = sqrt(5313)*(sqrt(23023)*sh_10_11*x + sqrt(10626)*sh_10_13*x - sqrt(10626)*sh_10_7*z + sqrt(69069)*sh_10_8*y + sqrt(23023)*sh_10_9*z)/19481
    sh_11_10 = sqrt(66)*sh_10_10*x/11 + 3*sqrt(10)*sh_10_12*x/22 - 3*sqrt(10)*sh_10_8*z/22 + 2*sqrt(30)*sh_10_9*y/11
    sh_11_11 = sh_10_10*y - sqrt(55)*sh_10_11*z/11 - sqrt(55)*sh_10_9*x/11
    sh_11_12 = sqrt(66)*sh_10_10*z/11 + 2*sqrt(30)*sh_10_11*y/11 - 3*sqrt(10)*sh_10_12*z/22 - 3*sqrt(10)*sh_10_8*x/22
    sh_11_13 = sqrt(5313)*(sqrt(23023)*sh_10_11*z + sqrt(69069)*sh_10_12*y - sqrt(10626)*sh_10_13*z - sqrt(10626)*sh_10_7*x - sqrt(23023)*sh_10_9*x)/19481
    sh_11_14 = sqrt(182)*sh_10_12*z/22 + 4*sqrt(7)*sh_10_13*y/11 - sqrt(14)*sh_10_14*z/11 - sqrt(14)*sh_10_6*x/11 - sqrt(182)*sh_10_8*x/22
    sh_11_15 = sqrt(210)*sh_10_13*z/22 + sqrt(105)*sh_10_14*y/11 - sqrt(42)*sh_10_15*z/22 - sqrt(42)*sh_10_5*x/22 - sqrt(210)*sh_10_7*x/22
    sh_11_16 = 2*sqrt(15)*sh_10_14*z/11 + 4*sqrt(6)*sh_10_15*y/11 - sqrt(30)*sh_10_16*z/22 - sqrt(30)*sh_10_4*x/22 - 2*sqrt(15)*sh_10_6*x/11
    sh_11_17 = 2*sqrt(17)*sh_10_15*z/11 + sqrt(85)*sh_10_16*y/11 - sqrt(5)*sh_10_17*z/11 - sqrt(5)*sh_10_3*x/11 - 2*sqrt(17)*sh_10_5*x/11
    sh_11_18 = 3*sqrt(34)*sh_10_16*z/22 + 6*sqrt(2)*sh_10_17*y/11 - sqrt(3)*sh_10_18*z/11 - sqrt(3)*sh_10_2*x/11 - 3*sqrt(34)*sh_10_4*x/22
    sh_11_19 = -sqrt(6)*sh_10_1*x/22 + 3*sqrt(38)*sh_10_17*z/22 + sqrt(57)*sh_10_18*y/11 - sqrt(6)*sh_10_19*z/22 - 3*sqrt(38)*sh_10_3*x/22
    sh_11_20 = -sqrt(2)*sh_10_0*x/22 + sqrt(95)*sh_10_18*z/11 + 2*sqrt(10)*sh_10_19*y/11 - sqrt(95)*sh_10_2*x/11 - sqrt(2)*sh_10_20*z/22
    sh_11_21 = -sqrt(105)*sh_10_1*x/11 + sqrt(105)*sh_10_19*z/11 + sqrt(21)*sh_10_20*y/11
    sh_11_22 = sqrt(462)*(-sh_10_0*x + sh_10_20*z)/22
    sh_1_0 = sqrt(3) * sh_1_0
    sh_1_1 = sqrt(3) * sh_1_1
    sh_1_2 = sqrt(3) * sh_1_2
    sh_2_0 = sqrt(5) * sh_2_0
    sh_2_1 = sqrt(5) * sh_2_1
    sh_2_2 = sqrt(5) * sh_2_2
    sh_2_3 = sqrt(5) * sh_2_3
    sh_2_4 = sqrt(5) * sh_2_4
    sh_3_0 = sqrt(7) * sh_3_0
    sh_3_1 = sqrt(7) * sh_3_1
    sh_3_2 = sqrt(7) * sh_3_2
    sh_3_3 = sqrt(7) * sh_3_3
    sh_3_4 = sqrt(7) * sh_3_4
    sh_3_5 = sqrt(7) * sh_3_5
    sh_3_6 = sqrt(7) * sh_3_6
    sh_4_0 = sqrt(9) * sh_4_0
    sh_4_1 = sqrt(9) * sh_4_1
    sh_4_2 = sqrt(9) * sh_4_2
    sh_4_3 = sqrt(9) * sh_4_3
    sh_4_4 = sqrt(9) * sh_4_4
    sh_4_5 = sqrt(9) * sh_4_5
    sh_4_6 = sqrt(9) * sh_4_6
    sh_4_7 = sqrt(9) * sh_4_7
    sh_4_8 = sqrt(9) * sh_4_8
    sh_5_0 = sqrt(11) * sh_5_0
    sh_5_1 = sqrt(11) * sh_5_1
    sh_5_2 = sqrt(11) * sh_5_2
    sh_5_3 = sqrt(11) * sh_5_3
    sh_5_4 = sqrt(11) * sh_5_4
    sh_5_5 = sqrt(11) * sh_5_5
    sh_5_6 = sqrt(11) * sh_5_6
    sh_5_7 = sqrt(11) * sh_5_7
    sh_5_8 = sqrt(11) * sh_5_8
    sh_5_9 = sqrt(11) * sh_5_9
    sh_5_10 = sqrt(11) * sh_5_10
    sh_6_0 = sqrt(13) * sh_6_0
    sh_6_1 = sqrt(13) * sh_6_1
    sh_6_2 = sqrt(13) * sh_6_2
    sh_6_3 = sqrt(13) * sh_6_3
    sh_6_4 = sqrt(13) * sh_6_4
    sh_6_5 = sqrt(13) * sh_6_5
    sh_6_6 = sqrt(13) * sh_6_6
    sh_6_7 = sqrt(13) * sh_6_7
    sh_6_8 = sqrt(13) * sh_6_8
    sh_6_9 = sqrt(13) * sh_6_9
    sh_6_10 = sqrt(13) * sh_6_10
    sh_6_11 = sqrt(13) * sh_6_11
    sh_6_12 = sqrt(13) * sh_6_12
    sh_7_0 = sqrt(15) * sh_7_0
    sh_7_1 = sqrt(15) * sh_7_1
    sh_7_2 = sqrt(15) * sh_7_2
    sh_7_3 = sqrt(15) * sh_7_3
    sh_7_4 = sqrt(15) * sh_7_4
    sh_7_5 = sqrt(15) * sh_7_5
    sh_7_6 = sqrt(15) * sh_7_6
    sh_7_7 = sqrt(15) * sh_7_7
    sh_7_8 = sqrt(15) * sh_7_8
    sh_7_9 = sqrt(15) * sh_7_9
    sh_7_10 = sqrt(15) * sh_7_10
    sh_7_11 = sqrt(15) * sh_7_11
    sh_7_12 = sqrt(15) * sh_7_12
    sh_7_13 = sqrt(15) * sh_7_13
    sh_7_14 = sqrt(15) * sh_7_14
    sh_8_0 = sqrt(17) * sh_8_0
    sh_8_1 = sqrt(17) * sh_8_1
    sh_8_2 = sqrt(17) * sh_8_2
    sh_8_3 = sqrt(17) * sh_8_3
    sh_8_4 = sqrt(17) * sh_8_4
    sh_8_5 = sqrt(17) * sh_8_5
    sh_8_6 = sqrt(17) * sh_8_6
    sh_8_7 = sqrt(17) * sh_8_7
    sh_8_8 = sqrt(17) * sh_8_8
    sh_8_9 = sqrt(17) * sh_8_9
    sh_8_10 = sqrt(17) * sh_8_10
    sh_8_11 = sqrt(17) * sh_8_11
    sh_8_12 = sqrt(17) * sh_8_12
    sh_8_13 = sqrt(17) * sh_8_13
    sh_8_14 = sqrt(17) * sh_8_14
    sh_8_15 = sqrt(17) * sh_8_15
    sh_8_16 = sqrt(17) * sh_8_16
    sh_9_0 = sqrt(19) * sh_9_0
    sh_9_1 = sqrt(19) * sh_9_1
    sh_9_2 = sqrt(19) * sh_9_2
    sh_9_3 = sqrt(19) * sh_9_3
    sh_9_4 = sqrt(19) * sh_9_4
    sh_9_5 = sqrt(19) * sh_9_5
    sh_9_6 = sqrt(19) * sh_9_6
    sh_9_7 = sqrt(19) * sh_9_7
    sh_9_8 = sqrt(19) * sh_9_8
    sh_9_9 = sqrt(19) * sh_9_9
    sh_9_10 = sqrt(19) * sh_9_10
    sh_9_11 = sqrt(19) * sh_9_11
    sh_9_12 = sqrt(19) * sh_9_12
    sh_9_13 = sqrt(19) * sh_9_13
    sh_9_14 = sqrt(19) * sh_9_14
    sh_9_15 = sqrt(19) * sh_9_15
    sh_9_16 = sqrt(19) * sh_9_16
    sh_9_17 = sqrt(19) * sh_9_17
    sh_9_18 = sqrt(19) * sh_9_18
    sh_10_0 = sqrt(21) * sh_10_0
    sh_10_1 = sqrt(21) * sh_10_1
    sh_10_2 = sqrt(21) * sh_10_2
    sh_10_3 = sqrt(21) * sh_10_3
    sh_10_4 = sqrt(21) * sh_10_4
    sh_10_5 = sqrt(21) * sh_10_5
    sh_10_6 = sqrt(21) * sh_10_6
    sh_10_7 = sqrt(21) * sh_10_7
    sh_10_8 = sqrt(21) * sh_10_8
    sh_10_9 = sqrt(21) * sh_10_9
    sh_10_10 = sqrt(21) * sh_10_10
    sh_10_11 = sqrt(21) * sh_10_11
    sh_10_12 = sqrt(21) * sh_10_12
    sh_10_13 = sqrt(21) * sh_10_13
    sh_10_14 = sqrt(21) * sh_10_14
    sh_10_15 = sqrt(21) * sh_10_15
    sh_10_16 = sqrt(21) * sh_10_16
    sh_10_17 = sqrt(21) * sh_10_17
    sh_10_18 = sqrt(21) * sh_10_18
    sh_10_19 = sqrt(21) * sh_10_19
    sh_10_20 = sqrt(21) * sh_10_20
    sh_11_0 = sqrt(23) * sh_11_0
    sh_11_1 = sqrt(23) * sh_11_1
    sh_11_2 = sqrt(23) * sh_11_2
    sh_11_3 = sqrt(23) * sh_11_3
    sh_11_4 = sqrt(23) * sh_11_4
    sh_11_5 = sqrt(23) * sh_11_5
    sh_11_6 = sqrt(23) * sh_11_6
    sh_11_7 = sqrt(23) * sh_11_7
    sh_11_8 = sqrt(23) * sh_11_8
    sh_11_9 = sqrt(23) * sh_11_9
    sh_11_10 = sqrt(23) * sh_11_10
    sh_11_11 = sqrt(23) * sh_11_11
    sh_11_12 = sqrt(23) * sh_11_12
    sh_11_13 = sqrt(23) * sh_11_13
    sh_11_14 = sqrt(23) * sh_11_14
    sh_11_15 = sqrt(23) * sh_11_15
    sh_11_16 = sqrt(23) * sh_11_16
    sh_11_17 = sqrt(23) * sh_11_17
    sh_11_18 = sqrt(23) * sh_11_18
    sh_11_19 = sqrt(23) * sh_11_19
    sh_11_20 = sqrt(23) * sh_11_20
    sh_11_21 = sqrt(23) * sh_11_21
    sh_11_22 = sqrt(23) * sh_11_22
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


@torch.jit.script
def _sph_lmax_0_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    return torch.stack([
        sh_0_0
    ], dim=-1)


@torch.jit.script
def _sph_lmax_1_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2
    ], dim=-1)


@torch.jit.script
def _sph_lmax_2_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4
    ], dim=-1)


@torch.jit.script
def _sph_lmax_3_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6
    ], dim=-1)


@torch.jit.script
def _sph_lmax_4_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8
    ], dim=-1)


@torch.jit.script
def _sph_lmax_5_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10
    ], dim=-1)


@torch.jit.script
def _sph_lmax_6_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    return torch.stack([
        sh_0_0,
        sh_1_0, sh_1_1, sh_1_2,
        sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
        sh_3_0, sh_3_1, sh_3_2, sh_3_3, sh_3_4, sh_3_5, sh_3_6,
        sh_4_0, sh_4_1, sh_4_2, sh_4_3, sh_4_4, sh_4_5, sh_4_6, sh_4_7, sh_4_8,
        sh_5_0, sh_5_1, sh_5_2, sh_5_3, sh_5_4, sh_5_5, sh_5_6, sh_5_7, sh_5_8, sh_5_9, sh_5_10,
        sh_6_0, sh_6_1, sh_6_2, sh_6_3, sh_6_4, sh_6_5, sh_6_6, sh_6_7, sh_6_8, sh_6_9, sh_6_10, sh_6_11, sh_6_12
    ], dim=-1)


@torch.jit.script
def _sph_lmax_7_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
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


@torch.jit.script
def _sph_lmax_8_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
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


@torch.jit.script
def _sph_lmax_9_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
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


@torch.jit.script
def _sph_lmax_10_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
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


@torch.jit.script
def _sph_lmax_11_norm(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    sqrt = torch.sqrt
    pi = 3.141592653589793
    sh_0_0 = torch.ones_like(x)
    sh_1_0 = x
    sh_1_1 = y
    sh_1_2 = z
    sh_2_0 = sqrt(3)*(sh_1_0*z + sh_1_2*x)/2
    sh_2_1 = sqrt(3)*(sh_1_0*y + sh_1_1*x)/2
    sh_2_2 = -sh_1_0*x/2 + sh_1_1*y - sh_1_2*z/2
    sh_2_3 = sqrt(3)*(sh_1_1*z + sh_1_2*y)/2
    sh_2_4 = sqrt(3)*(-sh_1_0*x + sh_1_2*z)/2
    sh_3_0 = sqrt(30)*(sh_2_0*z + sh_2_4*x)/6
    sh_3_1 = sqrt(5)*(sh_2_0*y + sh_2_1*z + sh_2_3*x)/3
    sh_3_2 = -sqrt(2)*sh_2_0*z/6 + 2*sqrt(2)*sh_2_1*y/3 + sqrt(6)*sh_2_2*x/3 + sqrt(2)*sh_2_4*x/6
    sh_3_3 = -sqrt(3)*sh_2_1*x/3 + sh_2_2*y - sqrt(3)*sh_2_3*z/3
    sh_3_4 = -sqrt(2)*sh_2_0*x/6 + sqrt(6)*sh_2_2*z/3 + 2*sqrt(2)*sh_2_3*y/3 - sqrt(2)*sh_2_4*z/6
    sh_3_5 = sqrt(5)*(-sh_2_1*x + sh_2_3*z + sh_2_4*y)/3
    sh_3_6 = sqrt(30)*(-sh_2_0*x + sh_2_4*z)/6
    sh_4_0 = sqrt(14)*(sh_3_0*z + sh_3_6*x)/4
    sh_4_1 = sqrt(7)*(2*sh_3_0*y + sqrt(6)*sh_3_1*z + sqrt(6)*sh_3_5*x)/8
    sh_4_2 = -sqrt(2)*sh_3_0*z/8 + sqrt(3)*sh_3_1*y/2 + sqrt(30)*sh_3_2*z/8 + sqrt(30)*sh_3_4*x/8 + sqrt(2)*sh_3_6*x/8
    sh_4_3 = -sqrt(6)*sh_3_1*z/8 + sqrt(15)*sh_3_2*y/4 + sqrt(10)*sh_3_3*x/4 + sqrt(6)*sh_3_5*x/8
    sh_4_4 = -sqrt(6)*sh_3_2*x/4 + sh_3_3*y - sqrt(6)*sh_3_4*z/4
    sh_4_5 = -sqrt(6)*sh_3_1*x/8 + sqrt(10)*sh_3_3*z/4 + sqrt(15)*sh_3_4*y/4 - sqrt(6)*sh_3_5*z/8
    sh_4_6 = -sqrt(2)*sh_3_0*x/8 - sqrt(30)*sh_3_2*x/8 + sqrt(30)*sh_3_4*z/8 + sqrt(3)*sh_3_5*y/2 - sqrt(2)*sh_3_6*z/8
    sh_4_7 = sqrt(7)*(-sqrt(6)*sh_3_1*x + sqrt(6)*sh_3_5*z + 2*sh_3_6*y)/8
    sh_4_8 = sqrt(14)*(-sh_3_0*x + sh_3_6*z)/4
    sh_5_0 = 3*sqrt(10)*(sh_4_0*z + sh_4_8*x)/10
    sh_5_1 = 3*sh_4_0*y/5 + 3*sqrt(2)*sh_4_1*z/5 + 3*sqrt(2)*sh_4_7*x/5
    sh_5_2 = -sqrt(2)*sh_4_0*z/10 + 4*sh_4_1*y/5 + sqrt(14)*sh_4_2*z/5 + sqrt(14)*sh_4_6*x/5 + sqrt(2)*sh_4_8*x/10
    sh_5_3 = -sqrt(6)*sh_4_1*z/10 + sqrt(21)*sh_4_2*y/5 + sqrt(42)*sh_4_3*z/10 + sqrt(42)*sh_4_5*x/10 + sqrt(6)*sh_4_7*x/10
    sh_5_4 = -sqrt(3)*sh_4_2*z/5 + 2*sqrt(6)*sh_4_3*y/5 + sqrt(15)*sh_4_4*x/5 + sqrt(3)*sh_4_6*x/5
    sh_5_5 = -sqrt(10)*sh_4_3*x/5 + sh_4_4*y - sqrt(10)*sh_4_5*z/5
    sh_5_6 = -sqrt(3)*sh_4_2*x/5 + sqrt(15)*sh_4_4*z/5 + 2*sqrt(6)*sh_4_5*y/5 - sqrt(3)*sh_4_6*z/5
    sh_5_7 = -sqrt(6)*sh_4_1*x/10 - sqrt(42)*sh_4_3*x/10 + sqrt(42)*sh_4_5*z/10 + sqrt(21)*sh_4_6*y/5 - sqrt(6)*sh_4_7*z/10
    sh_5_8 = -sqrt(2)*sh_4_0*x/10 - sqrt(14)*sh_4_2*x/5 + sqrt(14)*sh_4_6*z/5 + 4*sh_4_7*y/5 - sqrt(2)*sh_4_8*z/10
    sh_5_9 = -3*sqrt(2)*sh_4_1*x/5 + 3*sqrt(2)*sh_4_7*z/5 + 3*sh_4_8*y/5
    sh_5_10 = 3*sqrt(10)*(-sh_4_0*x + sh_4_8*z)/10
    sh_6_0 = sqrt(33)*(sh_5_0*z + sh_5_10*x)/6
    sh_6_1 = sqrt(11)*sh_5_0*y/6 + sqrt(110)*sh_5_1*z/12 + sqrt(110)*sh_5_9*x/12
    sh_6_2 = -sqrt(2)*sh_5_0*z/12 + sqrt(5)*sh_5_1*y/3 + sqrt(2)*sh_5_10*x/12 + sqrt(10)*sh_5_2*z/4 + sqrt(10)*sh_5_8*x/4
    sh_6_3 = -sqrt(6)*sh_5_1*z/12 + sqrt(3)*sh_5_2*y/2 + sqrt(2)*sh_5_3*z/2 + sqrt(2)*sh_5_7*x/2 + sqrt(6)*sh_5_9*x/12
    sh_6_4 = -sqrt(3)*sh_5_2*z/6 + 2*sqrt(2)*sh_5_3*y/3 + sqrt(14)*sh_5_4*z/6 + sqrt(14)*sh_5_6*x/6 + sqrt(3)*sh_5_8*x/6
    sh_6_5 = -sqrt(5)*sh_5_3*z/6 + sqrt(35)*sh_5_4*y/6 + sqrt(21)*sh_5_5*x/6 + sqrt(5)*sh_5_7*x/6
    sh_6_6 = -sqrt(15)*sh_5_4*x/6 + sh_5_5*y - sqrt(15)*sh_5_6*z/6
    sh_6_7 = -sqrt(5)*sh_5_3*x/6 + sqrt(21)*sh_5_5*z/6 + sqrt(35)*sh_5_6*y/6 - sqrt(5)*sh_5_7*z/6
    sh_6_8 = -sqrt(3)*sh_5_2*x/6 - sqrt(14)*sh_5_4*x/6 + sqrt(14)*sh_5_6*z/6 + 2*sqrt(2)*sh_5_7*y/3 - sqrt(3)*sh_5_8*z/6
    sh_6_9 = -sqrt(6)*sh_5_1*x/12 - sqrt(2)*sh_5_3*x/2 + sqrt(2)*sh_5_7*z/2 + sqrt(3)*sh_5_8*y/2 - sqrt(6)*sh_5_9*z/12
    sh_6_10 = -sqrt(2)*sh_5_0*x/12 - sqrt(2)*sh_5_10*z/12 - sqrt(10)*sh_5_2*x/4 + sqrt(10)*sh_5_8*z/4 + sqrt(5)*sh_5_9*y/3
    sh_6_11 = -sqrt(110)*sh_5_1*x/12 + sqrt(11)*sh_5_10*y/6 + sqrt(110)*sh_5_9*z/12
    sh_6_12 = sqrt(33)*(-sh_5_0*x + sh_5_10*z)/6
    sh_7_0 = sqrt(182)*(sh_6_0*z + sh_6_12*x)/14
    sh_7_1 = sqrt(13)*sh_6_0*y/7 + sqrt(39)*sh_6_1*z/7 + sqrt(39)*sh_6_11*x/7
    sh_7_2 = -sqrt(2)*sh_6_0*z/14 + 2*sqrt(6)*sh_6_1*y/7 + sqrt(33)*sh_6_10*x/7 + sqrt(2)*sh_6_12*x/14 + sqrt(33)*sh_6_2*z/7
    sh_7_3 = -sqrt(6)*sh_6_1*z/14 + sqrt(6)*sh_6_11*x/14 + sqrt(33)*sh_6_2*y/7 + sqrt(110)*sh_6_3*z/14 + sqrt(110)*sh_6_9*x/14
    sh_7_4 = sqrt(3)*sh_6_10*x/7 - sqrt(3)*sh_6_2*z/7 + 2*sqrt(10)*sh_6_3*y/7 + 3*sqrt(10)*sh_6_4*z/14 + 3*sqrt(10)*sh_6_8*x/14
    sh_7_5 = -sqrt(5)*sh_6_3*z/7 + 3*sqrt(5)*sh_6_4*y/7 + 3*sqrt(2)*sh_6_5*z/7 + 3*sqrt(2)*sh_6_7*x/7 + sqrt(5)*sh_6_9*x/7
    sh_7_6 = -sqrt(30)*sh_6_4*z/14 + 4*sqrt(3)*sh_6_5*y/7 + 2*sqrt(7)*sh_6_6*x/7 + sqrt(30)*sh_6_8*x/14
    sh_7_7 = -sqrt(21)*sh_6_5*x/7 + sh_6_6*y - sqrt(21)*sh_6_7*z/7
    sh_7_8 = -sqrt(30)*sh_6_4*x/14 + 2*sqrt(7)*sh_6_6*z/7 + 4*sqrt(3)*sh_6_7*y/7 - sqrt(30)*sh_6_8*z/14
    sh_7_9 = -sqrt(5)*sh_6_3*x/7 - 3*sqrt(2)*sh_6_5*x/7 + 3*sqrt(2)*sh_6_7*z/7 + 3*sqrt(5)*sh_6_8*y/7 - sqrt(5)*sh_6_9*z/7
    sh_7_10 = -sqrt(3)*sh_6_10*z/7 - sqrt(3)*sh_6_2*x/7 - 3*sqrt(10)*sh_6_4*x/14 + 3*sqrt(10)*sh_6_8*z/14 + 2*sqrt(10)*sh_6_9*y/7
    sh_7_11 = -sqrt(6)*sh_6_1*x/14 + sqrt(33)*sh_6_10*y/7 - sqrt(6)*sh_6_11*z/14 - sqrt(110)*sh_6_3*x/14 + sqrt(110)*sh_6_9*z/14
    sh_7_12 = -sqrt(2)*sh_6_0*x/14 + sqrt(33)*sh_6_10*z/7 + 2*sqrt(6)*sh_6_11*y/7 - sqrt(2)*sh_6_12*z/14 - sqrt(33)*sh_6_2*x/7
    sh_7_13 = -sqrt(39)*sh_6_1*x/7 + sqrt(39)*sh_6_11*z/7 + sqrt(13)*sh_6_12*y/7
    sh_7_14 = sqrt(182)*(-sh_6_0*x + sh_6_12*z)/14
    sh_8_0 = sqrt(15)*(sh_7_0*z + sh_7_14*x)/4
    sh_8_1 = sqrt(15)*sh_7_0*y/8 + sqrt(210)*sh_7_1*z/16 + sqrt(210)*sh_7_13*x/16
    sh_8_2 = -sqrt(2)*sh_7_0*z/16 + sqrt(7)*sh_7_1*y/4 + sqrt(182)*sh_7_12*x/16 + sqrt(2)*sh_7_14*x/16 + sqrt(182)*sh_7_2*z/16
    sh_8_3 = sqrt(510)*(-sqrt(85)*sh_7_1*z + sqrt(2210)*sh_7_11*x + sqrt(85)*sh_7_13*x + sqrt(2210)*sh_7_2*y + sqrt(2210)*sh_7_3*z)/1360
    sh_8_4 = sqrt(33)*sh_7_10*x/8 + sqrt(3)*sh_7_12*x/8 - sqrt(3)*sh_7_2*z/8 + sqrt(3)*sh_7_3*y/2 + sqrt(33)*sh_7_4*z/8
    sh_8_5 = sqrt(510)*(sqrt(102)*sh_7_11*x - sqrt(102)*sh_7_3*z + sqrt(1122)*sh_7_4*y + sqrt(561)*sh_7_5*z + sqrt(561)*sh_7_9*x)/816
    sh_8_6 = sqrt(30)*sh_7_10*x/16 - sqrt(30)*sh_7_4*z/16 + sqrt(15)*sh_7_5*y/4 + 3*sqrt(10)*sh_7_6*z/16 + 3*sqrt(10)*sh_7_8*x/16
    sh_8_7 = -sqrt(42)*sh_7_5*z/16 + 3*sqrt(7)*sh_7_6*y/8 + 3*sh_7_7*x/4 + sqrt(42)*sh_7_9*x/16
    sh_8_8 = -sqrt(7)*sh_7_6*x/4 + sh_7_7*y - sqrt(7)*sh_7_8*z/4
    sh_8_9 = -sqrt(42)*sh_7_5*x/16 + 3*sh_7_7*z/4 + 3*sqrt(7)*sh_7_8*y/8 - sqrt(42)*sh_7_9*z/16
    sh_8_10 = -sqrt(30)*sh_7_10*z/16 - sqrt(30)*sh_7_4*x/16 - 3*sqrt(10)*sh_7_6*x/16 + 3*sqrt(10)*sh_7_8*z/16 + sqrt(15)*sh_7_9*y/4
    sh_8_11 = sqrt(510)*(sqrt(1122)*sh_7_10*y - sqrt(102)*sh_7_11*z - sqrt(102)*sh_7_3*x - sqrt(561)*sh_7_5*x + sqrt(561)*sh_7_9*z)/816
    sh_8_12 = sqrt(33)*sh_7_10*z/8 + sqrt(3)*sh_7_11*y/2 - sqrt(3)*sh_7_12*z/8 - sqrt(3)*sh_7_2*x/8 - sqrt(33)*sh_7_4*x/8
    sh_8_13 = sqrt(510)*(-sqrt(85)*sh_7_1*x + sqrt(2210)*sh_7_11*z + sqrt(2210)*sh_7_12*y - sqrt(85)*sh_7_13*z - sqrt(2210)*sh_7_3*x)/1360
    sh_8_14 = -sqrt(2)*sh_7_0*x/16 + sqrt(182)*sh_7_12*z/16 + sqrt(7)*sh_7_13*y/4 - sqrt(2)*sh_7_14*z/16 - sqrt(182)*sh_7_2*x/16
    sh_8_15 = -sqrt(210)*sh_7_1*x/16 + sqrt(210)*sh_7_13*z/16 + sqrt(15)*sh_7_14*y/8
    sh_8_16 = sqrt(15)*(-sh_7_0*x + sh_7_14*z)/4
    sh_9_0 = sqrt(34)*(sh_8_0*z + sh_8_16*x)/6
    sh_9_1 = sqrt(17)*(sh_8_0*y + 2*sh_8_1*z + 2*sh_8_15*x)/9
    sh_9_2 = -sqrt(2)*sh_8_0*z/18 + 4*sqrt(2)*sh_8_1*y/9 + 2*sqrt(15)*sh_8_14*x/9 + sqrt(2)*sh_8_16*x/18 + 2*sqrt(15)*sh_8_2*z/9
    sh_9_3 = -sqrt(6)*sh_8_1*z/18 + sqrt(210)*sh_8_13*x/18 + sqrt(6)*sh_8_15*x/18 + sqrt(5)*sh_8_2*y/3 + sqrt(210)*sh_8_3*z/18
    sh_9_4 = sqrt(182)*sh_8_12*x/18 + sqrt(3)*sh_8_14*x/9 - sqrt(3)*sh_8_2*z/9 + 2*sqrt(14)*sh_8_3*y/9 + sqrt(182)*sh_8_4*z/18
    sh_9_5 = sqrt(39)*sh_8_11*x/9 + sqrt(5)*sh_8_13*x/9 - sqrt(5)*sh_8_3*z/9 + sqrt(65)*sh_8_4*y/9 + sqrt(39)*sh_8_5*z/9
    sh_9_6 = sqrt(33)*sh_8_10*x/9 + sqrt(30)*sh_8_12*x/18 - sqrt(30)*sh_8_4*z/18 + 2*sqrt(2)*sh_8_5*y/3 + sqrt(33)*sh_8_6*z/9
    sh_9_7 = sqrt(42)*sh_8_11*x/18 - sqrt(42)*sh_8_5*z/18 + sqrt(77)*sh_8_6*y/9 + sqrt(110)*sh_8_7*z/18 + sqrt(110)*sh_8_9*x/18
    sh_9_8 = sqrt(14)*sh_8_10*x/9 - sqrt(14)*sh_8_6*z/9 + 4*sqrt(5)*sh_8_7*y/9 + sqrt(5)*sh_8_8*x/3
    sh_9_9 = -2*sh_8_7*x/3 + sh_8_8*y - 2*sh_8_9*z/3
    sh_9_10 = -sqrt(14)*sh_8_10*z/9 - sqrt(14)*sh_8_6*x/9 + sqrt(5)*sh_8_8*z/3 + 4*sqrt(5)*sh_8_9*y/9
    sh_9_11 = sqrt(77)*sh_8_10*y/9 - sqrt(42)*sh_8_11*z/18 - sqrt(42)*sh_8_5*x/18 - sqrt(110)*sh_8_7*x/18 + sqrt(110)*sh_8_9*z/18
    sh_9_12 = sqrt(33)*sh_8_10*z/9 + 2*sqrt(2)*sh_8_11*y/3 - sqrt(30)*sh_8_12*z/18 - sqrt(30)*sh_8_4*x/18 - sqrt(33)*sh_8_6*x/9
    sh_9_13 = sqrt(39)*sh_8_11*z/9 + sqrt(65)*sh_8_12*y/9 - sqrt(5)*sh_8_13*z/9 - sqrt(5)*sh_8_3*x/9 - sqrt(39)*sh_8_5*x/9
    sh_9_14 = sqrt(182)*sh_8_12*z/18 + 2*sqrt(14)*sh_8_13*y/9 - sqrt(3)*sh_8_14*z/9 - sqrt(3)*sh_8_2*x/9 - sqrt(182)*sh_8_4*x/18
    sh_9_15 = -sqrt(6)*sh_8_1*x/18 + sqrt(210)*sh_8_13*z/18 + sqrt(5)*sh_8_14*y/3 - sqrt(6)*sh_8_15*z/18 - sqrt(210)*sh_8_3*x/18
    sh_9_16 = -sqrt(2)*sh_8_0*x/18 + 2*sqrt(15)*sh_8_14*z/9 + 4*sqrt(2)*sh_8_15*y/9 - sqrt(2)*sh_8_16*z/18 - 2*sqrt(15)*sh_8_2*x/9
    sh_9_17 = sqrt(17)*(-2*sh_8_1*x + 2*sh_8_15*z + sh_8_16*y)/9
    sh_9_18 = sqrt(34)*(-sh_8_0*x + sh_8_16*z)/6
    sh_10_0 = sqrt(95)*(sh_9_0*z + sh_9_18*x)/10
    sh_10_1 = sqrt(19)*sh_9_0*y/10 + 3*sqrt(38)*sh_9_1*z/20 + 3*sqrt(38)*sh_9_17*x/20
    sh_10_2 = -sqrt(2)*sh_9_0*z/20 + 3*sh_9_1*y/5 + 3*sqrt(34)*sh_9_16*x/20 + sqrt(2)*sh_9_18*x/20 + 3*sqrt(34)*sh_9_2*z/20
    sh_10_3 = -sqrt(6)*sh_9_1*z/20 + sqrt(17)*sh_9_15*x/5 + sqrt(6)*sh_9_17*x/20 + sqrt(51)*sh_9_2*y/10 + sqrt(17)*sh_9_3*z/5
    sh_10_4 = sqrt(15)*sh_9_14*x/5 + sqrt(3)*sh_9_16*x/10 - sqrt(3)*sh_9_2*z/10 + 4*sh_9_3*y/5 + sqrt(15)*sh_9_4*z/5
    sh_10_5 = sqrt(210)*sh_9_13*x/20 + sqrt(5)*sh_9_15*x/10 - sqrt(5)*sh_9_3*z/10 + sqrt(3)*sh_9_4*y/2 + sqrt(210)*sh_9_5*z/20
    sh_10_6 = sqrt(182)*sh_9_12*x/20 + sqrt(30)*sh_9_14*x/20 - sqrt(30)*sh_9_4*z/20 + sqrt(21)*sh_9_5*y/5 + sqrt(182)*sh_9_6*z/20
    sh_10_7 = sqrt(39)*sh_9_11*x/10 + sqrt(42)*sh_9_13*x/20 - sqrt(42)*sh_9_5*z/20 + sqrt(91)*sh_9_6*y/10 + sqrt(39)*sh_9_7*z/10
    sh_10_8 = sqrt(33)*sh_9_10*x/10 + sqrt(14)*sh_9_12*x/10 - sqrt(14)*sh_9_6*z/10 + 2*sqrt(6)*sh_9_7*y/5 + sqrt(33)*sh_9_8*z/10
    sh_10_9 = 3*sqrt(2)*sh_9_11*x/10 - 3*sqrt(2)*sh_9_7*z/10 + 3*sqrt(11)*sh_9_8*y/10 + sqrt(55)*sh_9_9*x/10
    sh_10_10 = -3*sqrt(5)*sh_9_10*z/10 - 3*sqrt(5)*sh_9_8*x/10 + sh_9_9*y
    sh_10_11 = 3*sqrt(11)*sh_9_10*y/10 - 3*sqrt(2)*sh_9_11*z/10 - 3*sqrt(2)*sh_9_7*x/10 + sqrt(55)*sh_9_9*z/10
    sh_10_12 = sqrt(33)*sh_9_10*z/10 + 2*sqrt(6)*sh_9_11*y/5 - sqrt(14)*sh_9_12*z/10 - sqrt(14)*sh_9_6*x/10 - sqrt(33)*sh_9_8*x/10
    sh_10_13 = sqrt(39)*sh_9_11*z/10 + sqrt(91)*sh_9_12*y/10 - sqrt(42)*sh_9_13*z/20 - sqrt(42)*sh_9_5*x/20 - sqrt(39)*sh_9_7*x/10
    sh_10_14 = sqrt(182)*sh_9_12*z/20 + sqrt(21)*sh_9_13*y/5 - sqrt(30)*sh_9_14*z/20 - sqrt(30)*sh_9_4*x/20 - sqrt(182)*sh_9_6*x/20
    sh_10_15 = sqrt(210)*sh_9_13*z/20 + sqrt(3)*sh_9_14*y/2 - sqrt(5)*sh_9_15*z/10 - sqrt(5)*sh_9_3*x/10 - sqrt(210)*sh_9_5*x/20
    sh_10_16 = sqrt(15)*sh_9_14*z/5 + 4*sh_9_15*y/5 - sqrt(3)*sh_9_16*z/10 - sqrt(3)*sh_9_2*x/10 - sqrt(15)*sh_9_4*x/5
    sh_10_17 = -sqrt(6)*sh_9_1*x/20 + sqrt(17)*sh_9_15*z/5 + sqrt(51)*sh_9_16*y/10 - sqrt(6)*sh_9_17*z/20 - sqrt(17)*sh_9_3*x/5
    sh_10_18 = -sqrt(2)*sh_9_0*x/20 + 3*sqrt(34)*sh_9_16*z/20 + 3*sh_9_17*y/5 - sqrt(2)*sh_9_18*z/20 - 3*sqrt(34)*sh_9_2*x/20
    sh_10_19 = -3*sqrt(38)*sh_9_1*x/20 + 3*sqrt(38)*sh_9_17*z/20 + sqrt(19)*sh_9_18*y/10
    sh_10_20 = sqrt(95)*(-sh_9_0*x + sh_9_18*z)/10
    sh_11_0 = sqrt(462)*(sh_10_0*z + sh_10_20*x)/22
    sh_11_1 = sqrt(21)*sh_10_0*y/11 + sqrt(105)*sh_10_1*z/11 + sqrt(105)*sh_10_19*x/11
    sh_11_2 = -sqrt(2)*sh_10_0*z/22 + 2*sqrt(10)*sh_10_1*y/11 + sqrt(95)*sh_10_18*x/11 + sqrt(95)*sh_10_2*z/11 + sqrt(2)*sh_10_20*x/22
    sh_11_3 = -sqrt(6)*sh_10_1*z/22 + 3*sqrt(38)*sh_10_17*x/22 + sqrt(6)*sh_10_19*x/22 + sqrt(57)*sh_10_2*y/11 + 3*sqrt(38)*sh_10_3*z/22
    sh_11_4 = 3*sqrt(34)*sh_10_16*x/22 + sqrt(3)*sh_10_18*x/11 - sqrt(3)*sh_10_2*z/11 + 6*sqrt(2)*sh_10_3*y/11 + 3*sqrt(34)*sh_10_4*z/22
    sh_11_5 = 2*sqrt(17)*sh_10_15*x/11 + sqrt(5)*sh_10_17*x/11 - sqrt(5)*sh_10_3*z/11 + sqrt(85)*sh_10_4*y/11 + 2*sqrt(17)*sh_10_5*z/11
    sh_11_6 = 2*sqrt(15)*sh_10_14*x/11 + sqrt(30)*sh_10_16*x/22 - sqrt(30)*sh_10_4*z/22 + 4*sqrt(6)*sh_10_5*y/11 + 2*sqrt(15)*sh_10_6*z/11
    sh_11_7 = sqrt(210)*sh_10_13*x/22 + sqrt(42)*sh_10_15*x/22 - sqrt(42)*sh_10_5*z/22 + sqrt(105)*sh_10_6*y/11 + sqrt(210)*sh_10_7*z/22
    sh_11_8 = sqrt(182)*sh_10_12*x/22 + sqrt(14)*sh_10_14*x/11 - sqrt(14)*sh_10_6*z/11 + 4*sqrt(7)*sh_10_7*y/11 + sqrt(182)*sh_10_8*z/22
    sh_11_9 = sqrt(5313)*(sqrt(23023)*sh_10_11*x + sqrt(10626)*sh_10_13*x - sqrt(10626)*sh_10_7*z + sqrt(69069)*sh_10_8*y + sqrt(23023)*sh_10_9*z)/19481
    sh_11_10 = sqrt(66)*sh_10_10*x/11 + 3*sqrt(10)*sh_10_12*x/22 - 3*sqrt(10)*sh_10_8*z/22 + 2*sqrt(30)*sh_10_9*y/11
    sh_11_11 = sh_10_10*y - sqrt(55)*sh_10_11*z/11 - sqrt(55)*sh_10_9*x/11
    sh_11_12 = sqrt(66)*sh_10_10*z/11 + 2*sqrt(30)*sh_10_11*y/11 - 3*sqrt(10)*sh_10_12*z/22 - 3*sqrt(10)*sh_10_8*x/22
    sh_11_13 = sqrt(5313)*(sqrt(23023)*sh_10_11*z + sqrt(69069)*sh_10_12*y - sqrt(10626)*sh_10_13*z - sqrt(10626)*sh_10_7*x - sqrt(23023)*sh_10_9*x)/19481
    sh_11_14 = sqrt(182)*sh_10_12*z/22 + 4*sqrt(7)*sh_10_13*y/11 - sqrt(14)*sh_10_14*z/11 - sqrt(14)*sh_10_6*x/11 - sqrt(182)*sh_10_8*x/22
    sh_11_15 = sqrt(210)*sh_10_13*z/22 + sqrt(105)*sh_10_14*y/11 - sqrt(42)*sh_10_15*z/22 - sqrt(42)*sh_10_5*x/22 - sqrt(210)*sh_10_7*x/22
    sh_11_16 = 2*sqrt(15)*sh_10_14*z/11 + 4*sqrt(6)*sh_10_15*y/11 - sqrt(30)*sh_10_16*z/22 - sqrt(30)*sh_10_4*x/22 - 2*sqrt(15)*sh_10_6*x/11
    sh_11_17 = 2*sqrt(17)*sh_10_15*z/11 + sqrt(85)*sh_10_16*y/11 - sqrt(5)*sh_10_17*z/11 - sqrt(5)*sh_10_3*x/11 - 2*sqrt(17)*sh_10_5*x/11
    sh_11_18 = 3*sqrt(34)*sh_10_16*z/22 + 6*sqrt(2)*sh_10_17*y/11 - sqrt(3)*sh_10_18*z/11 - sqrt(3)*sh_10_2*x/11 - 3*sqrt(34)*sh_10_4*x/22
    sh_11_19 = -sqrt(6)*sh_10_1*x/22 + 3*sqrt(38)*sh_10_17*z/22 + sqrt(57)*sh_10_18*y/11 - sqrt(6)*sh_10_19*z/22 - 3*sqrt(38)*sh_10_3*x/22
    sh_11_20 = -sqrt(2)*sh_10_0*x/22 + sqrt(95)*sh_10_18*z/11 + 2*sqrt(10)*sh_10_19*y/11 - sqrt(95)*sh_10_2*x/11 - sqrt(2)*sh_10_20*z/22
    sh_11_21 = -sqrt(105)*sh_10_1*x/11 + sqrt(105)*sh_10_19*z/11 + sqrt(21)*sh_10_20*y/11
    sh_11_22 = sqrt(462)*(-sh_10_0*x + sh_10_20*z)/22
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


_spherical_harmonics = {(0, 'integral'): _sph_lmax_0_integral, (1, 'integral'): _sph_lmax_1_integral, (2, 'integral'): _sph_lmax_2_integral, (3, 'integral'): _sph_lmax_3_integral, (4, 'integral'): _sph_lmax_4_integral, (5, 'integral'): _sph_lmax_5_integral, (6, 'integral'): _sph_lmax_6_integral, (7, 'integral'): _sph_lmax_7_integral, (8, 'integral'): _sph_lmax_8_integral, (9, 'integral'): _sph_lmax_9_integral, (10, 'integral'): _sph_lmax_10_integral, (11, 'integral'): _sph_lmax_11_integral, (0, 'component'): _sph_lmax_0_component, (1, 'component'): _sph_lmax_1_component, (2, 'component'): _sph_lmax_2_component, (3, 'component'): _sph_lmax_3_component, (4, 'component'): _sph_lmax_4_component, (5, 'component'): _sph_lmax_5_component, (6, 'component'): _sph_lmax_6_component, (7, 'component'): _sph_lmax_7_component, (8, 'component'): _sph_lmax_8_component, (9, 'component'): _sph_lmax_9_component, (10, 'component'): _sph_lmax_10_component, (11, 'component'): _sph_lmax_11_component, (0, 'norm'): _sph_lmax_0_norm, (1, 'norm'): _sph_lmax_1_norm, (2, 'norm'): _sph_lmax_2_norm, (3, 'norm'): _sph_lmax_3_norm, (4, 'norm'): _sph_lmax_4_norm, (5, 'norm'): _sph_lmax_5_norm, (6, 'norm'): _sph_lmax_6_norm, (7, 'norm'): _sph_lmax_7_norm, (8, 'norm'): _sph_lmax_8_norm, (9, 'norm'): _sph_lmax_9_norm, (10, 'norm'): _sph_lmax_10_norm, (11, 'norm'): _sph_lmax_11_norm}
