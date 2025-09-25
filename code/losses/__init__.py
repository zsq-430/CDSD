from .constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

from .jaccard import JaccardLoss
from .dice import DiceLoss
from .focal import FocalLoss
# from .lovasz2 import LovaszLoss
# from .lovasz_losses import LovaszLoss
from .lovasz import LovaszLoss
from .soft_bce import SoftBCEWithLogitsLoss
from .soft_ce import SoftCrossEntropyLoss
from .tversky import TverskyLoss
from .tversky import TverskyLoss
from .myInfonce import MyInfoNCE
