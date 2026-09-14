import numpy as np
from numpy.typing import NDArray

from src.data_augmentors.simulation import DA_STD, NullSpaceTranslation


class ScaleTranslation(NullSpaceTranslation):
    """Translation along one direction: x -> x + c v-hat, c ~ N(0, (scale std)^2).

    Homogeneity of degree zero on the cigarette panel, where v = (1,1,1,1) and h_*
    is T-invariant iff v'b = 0. The parent takes a basis of the INVARIANT subspace
    and translates along its kernel; here the invariant subspace is null(v), so the
    kernel is v-hat itself and `param_dimension` is 1.

    Two knobs, as on the simulation. The call-time `scale` moves the amplitude
    along v and drives Prop. 2's lever; `strength` adds an isotropic component
    INSIDE null(v), which h_* is not invariant to -- misspecified symmetry, and the
    only channel the robustness sweep uses.
    """

    def __init__(self, direction: NDArray, std: float = DA_STD, strength: float = 0.0):
        direction = np.asarray(direction, dtype=float).ravel()
        # basis of null(v): the (k-1)-dimensional subspace h_* is invariant on
        super().__init__(W_XY=self.null_space(direction[None, :]), kernel_dim=-1, std=std, strength=strength)
        self.direction = direction / np.linalg.norm(direction)
        # UNIT rows. The parent normalises its basis in the Frobenius norm, which
        # for an orthonormal one puts a 1/sqrt(k-1) factor on `strength`: harmless,
        # since the knob is bisected against a target eps*, but wrong as documentation.
        self.W_perp = self.W_XY.T
