from .models import register, make
try:
    from . import ocr_rodosol
except Exception:
    ocr_rodosol = None
from . import ocr_adapters
from . import lpsrgan_arch
from . import GP_LPR_arch
