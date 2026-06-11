from recbole.utils import InputType

from .etrc_vug_crossfit_base import ETRCVUGCrossFitBase, _UserCrossFitTableMixin


class ETRCVUGUserCF(_UserCrossFitTableMixin, ETRCVUGCrossFitBase):
    input_type = InputType.POINTWISE
