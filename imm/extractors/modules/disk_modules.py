import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical


from .unet import thin_setup

DEFAULT_SETUP = {**thin_setup, "bias": True, "padding": True}


class Features:
    def __init__(self, kp, desc, kp_logp):
        assert kp.device == desc.device
        assert kp.device == kp_logp.device

        self.kp = kp
        self.desc = desc
        self.kp_logp = kp_logp

    @property
    def n(self):
        return self.kp.shape[0]

    @property
    def device(self):
        return self.kp.device

    def detached_and_grad_(self):
        return Features(
            self.kp,
            self.desc.detach().requires_grad_(),
            self.kp_logp.detach().requires_grad_(),
        )

    def requires_grad_(self, is_on):
        self.desc.requires_grad_(is_on)
        self.kp_logp.requires_grad_(is_on)

    def grad_tensors(self):
        return [self.desc, self.kp_logp]

    def to(self, *args, **kwargs):
        return Features(
            self.kp.to(*args, **kwargs),
            self.desc.to(*args, **kwargs),
            self.kp_logp.to(*args, **kwargs) if self.kp_logp is not None else None,
        )


def nms(signal, window_size=5, cutoff=0.0):
    if window_size % 2 != 1:
        raise ValueError(f"window_size has to be odd, got {window_size}")

    _, ixs = F.max_pool2d(
        signal,
        kernel_size=window_size,
        stride=1,
        padding=window_size // 2,
        return_indices=True,
    )

    # https://github.com/pytorch/pytorch/issues/38986
    # is fixed
    if len(ixs.shape) == 4:
        assert ixs.shape[0] == 1
        ixs = ixs.squeeze(0)

    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    nms = ixs == coords

    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)


def select_on_last(values, indices):
    """
    WARNING: this may be reinventing the wheel, but I don't know how to do
    it otherwise with PyTorch.
    This function uses an array of linear indices `indices` between [0, T] to
    index into `values` which has equal shape as `indices` and then one extra
    dimension of size T.
    """
    return torch.gather(values, -1, indices[..., None]).squeeze(-1)


def point_distribution(logits):
    """
    Implements the categorical proposal -> Bernoulli acceptance sampling
    scheme. Given a tensor of logits, performs samples on the last dimension,
    returning
        a) the proposals
        b) a binary mask indicating which ones were accepted
        c) the logp-probability of (proposal and acceptance decision)
    """

    proposal_dist = Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    accept_dist = Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample()
    accept_logp = accept_dist.log_prob(accept_samples)
    accept_mask = accept_samples == 1.0

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


class Keypoints:
    """
    A simple, temporary struct used to store keypoint detections and their
    log-probabilities. After construction, merge_with_descriptors is used to
    select corresponding descriptors from unet output.
    """

    def __init__(self, xys, logp):
        self.xys = xys
        self.logp = logp

    def merge_with_descriptors(self, descriptors):
        """
        Select descriptors from a dense `descriptors` tensor, at locations
        given by `self.xys`
        """
        x, y = self.xys.T

        desc = descriptors[:, y, x].T
        desc = F.normalize(desc, dim=-1)

        return Features(self.xys.to(torch.float32), desc, self.logp)


class Detector:
    def __init__(self, window=8):
        self.window = window

    def _tile(self, heatmap):
        """
        Divides the heatmap `heatmap` into tiles of size (v, v) where
        v==self.window. The tiles are flattened, resulting in the last
        dimension of the output T == v * v.
        """
        v = self.window
        b, c, h, w = heatmap.shape

        assert heatmap.shape[2] % v == 0
        assert heatmap.shape[3] % v == 0

        return heatmap.unfold(2, v, v).unfold(3, v, v).reshape(b, c, h // v, w // v, v * v)

    def sample(self, heatmap):
        """
        Implements the training-time grid-based sampling protocol
        """
        v = self.window
        dev = heatmap.device
        B, _, H, W = heatmap.shape

        assert H % v == 0
        assert W % v == 0

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(
            torch.meshgrid(
                torch.arange(H, device=dev),
                torch.arange(W, device=dev),
            )[::-1],
            dim=0,
        ).unsqueeze(0)
        self._tile(cgrid)

        # extract xy coordinates from cgrid according to indices sampled
        # before
        xys = select_on_last(
            self._tile(cgrid).repeat(B, 1, 1, 1, 1),
            # unsqueeze and repeat on the (xy) dimension to grab
            # both components from the grid
            proposals.unsqueeze(1).repeat(1, 2, 1, 1),
        ).permute(0, 2, 3, 1)  # -> bhw2

        keypoints = []
        for i in range(B):
            mask = accept_mask[i]
            keypoints.append(
                Keypoints(
                    xys[i][mask],
                    logp[i][mask],
                )
            )

        return np.array(keypoints, dtype=object)

    def nms(self, heatmap, n=None, **kwargs):
        """
        Inference-time nms-based detection protocol
        """
        heatmap = heatmap.squeeze(1)
        nmsed = nms(heatmap, **kwargs)

        keypoints = []
        for b in range(heatmap.shape[0]):
            yx = nmsed[b].nonzero(as_tuple=False)
            logp = heatmap[b][nmsed[b]]
            xy = torch.flip(yx, (1,))

            if n is not None:
                n_ = min(n + 1, logp.numel())
                # torch.kthvalue picks in ascending order and we want to pick in
                # descending order, so we pick n-th smallest among -logp to get
                # -threshold
                minus_threshold, _indices = torch.kthvalue(-logp, n_)
                mask = logp > -minus_threshold

                xy = xy[mask]
                logp = logp[mask]

            keypoints.append(Keypoints(xy, logp))

        return np.array(keypoints, dtype=object)
