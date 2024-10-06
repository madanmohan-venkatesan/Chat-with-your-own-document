from os import path
from llama_index.core import SimpleDirectoryReader
import pickle as pkl

CURRENT_PATH="."
DATA_PATH=path.join(CURRENT_PATH,"Data")
INTERIM_PATH=