from pathlib import Path
from urllib.parse import urljoin

from glados.config import Config


config = Config()

##############################################################################
# Edit values here, based on glados/config.py defaults

config.LLAMA_SERVER_EXTERNAL = True
config.LLAMA_SERVER_BASE_URL = "http://localhost:5000/"
