from recbole.utils import InputType

from .etrc_vug_crossfit_base import ETRCVUGCrossFitBase, _PairCrossFitTableMixin


class ETRCVUGPairCF(_PairCrossFitTableMixin, ETRCVUGCrossFitBase):
    input_type = InputType.POINTWISE
