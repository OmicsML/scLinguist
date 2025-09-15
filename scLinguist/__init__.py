import logging
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

from .model import model
from .data_loaders import data_loader
from .model import modeling_hyena

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("scLinguist: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

_all_ = ["scTrans", "HyenaConfig", "scMultiDataset", "paProteinDataset", 
         "paTESTProteinDataset_citeseq", "scRNADataset", "spRNADataset",
           "spMultiDataset", "scMultiDataset"]

# _version_ = version("scLinguist")