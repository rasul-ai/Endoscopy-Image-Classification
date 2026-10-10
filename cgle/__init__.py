from .model import LAG, CGLE, CrossAttention
from .ablations import GlobalOnly, LocalOnly, LAGNoCA

MODELS = {
    "cgle": LAG,
    "global_only": GlobalOnly,
    "local_only": LocalOnly,
    "fusion_no_ca": LAGNoCA,
}
