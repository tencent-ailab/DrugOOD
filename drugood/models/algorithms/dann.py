# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.

from torch.autograd import Function

from drugood.models.algorithms.base import BaseAlgorithm
from ..builder import MODELS, build_tasker
from ..builder import build_head


class GradientReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


@MODELS.register_module()
class DANN(BaseAlgorithm):
    """
        DANN.
        This algorithm was originally proposed as an unsupervised domain adaptation algorithm.
        Original paper:
            @article{ganin2016domain,
              title={Domain-adversarial training of neural networks},
              author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle,
               Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
              journal={The journal of machine learning research},
              volume={17},
              number={1},
              pages={2096--2030},
              year={2016},
              publisher={JMLR. org}
            }
    """

    def __init__(self, tasker, dann_cfg=None):
        super().__init__()
        self.tasker = build_tasker(tasker)
        assert dann_cfg is not None
        self.alpha = dann_cfg.get("alpha")
        self.aux_head = build_head(dann_cfg.get("aux_head"))

    def init_weights(self):
        pass

    def encode(self, input, **kwargs):
        feats = self.tasker.extract_feat(input, **kwargs)
        return feats

    def decode(self, feats, gt_label=None, return_loss=False):
        if return_loss:
            return self.tasker.head.forward_train(feats, gt_label)
        else:
            return self.tasker.head.forward_test(feats)

    def forward_train(self, input, group, gt_label, **kwargs):
        feats = self.encode(input, **kwargs)
        losses = self.decode(feats, gt_label, return_loss=True)

        _feature = GradientReverseLayerF.apply(feats, self.alpha)
        aux_losses = self.aux_head.forward_train(_feature, group)
        losses.update({f"aux_{key}": val for key, val in aux_losses.items()})
        return losses

    def simple_test(self, input, group, **kwargs):
        feats = self.encode(input, **kwargs)
        logits = self.decode(feats)
        preds = self.tasker.head.post_process(logits)
        return preds
