#For loading PDF
from os import path
from llama_index.core import SimpleDirectoryReader
import pickle as pkl

#For embedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import Settings

#For prompt template
from llama_index.core import PromptTemplate

#For downloading model
from huggingface_hub import notebook_login

#For storing embedded vectors in vector store
import torch

from transformers import AutoModelForCausalLM,AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM

from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

#For re-ranking
from llama_index.core.postprocessor import SentenceTransformerRerank

import time

#Paths
CURRENT_PATH="."
DATA_PATH=path.join(CURRENT_PATH,"Data")
INTERIM_PATH=path.join(CURRENT_PATH,"Interim")

#Load the PDF Document
documents = SimpleDirectoryReader("/content/Data").load_data()
embed_model =FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 512

#System prompt
system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

#Download model
notebook_login()


tokenizer = AutoTokenizer.from_pretrained(
"meta-llama/Llama-3.2-3B-Instruct"
)
stopping_ids = [
tokenizer.eos_token_id,
tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]
llm = HuggingFaceLLM(
context_window=8192,
max_new_tokens=256,
generate_kwargs={"temperature": 0.7, "do_sample":
False},
system_prompt=system_prompt,
query_wrapper_prompt=query_wrapper_prompt,
tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
model_name="meta-llama/Llama-3.2-3B-Instruct",
device_map="auto",
stopping_ids=stopping_ids,
tokenizer_kwargs={"max_length": 4096},
# uncomment this if using CUDA to reduce memoryusage
# model_kwargs={"torch_dtype": torch.float16}
)
Settings.llm = llm
Settings.chunk_size = 512

#Store embeddings in vector store
from IPython.display import Markdown, display
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
#
client = qdrant_client.QdrantClient(
# you can use :memory: mode for fast and light-weight experiments,
# it does not require to have Qdrant deployed anywhere
# but requires qdrant-client >= 1.1.1
location=":memory:"
# otherwise set Qdrant instance address with:
# url="http://<host>:<port>"
# otherwise set Qdrant instance with host and port:
#host="localhost",
#port=6333
# set API KEY for Qdrant Cloud
#api_key=<YOUR API KEY>
)

vector_store = QdrantVectorStore(client=client,collection_name="test")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents,storage_context=storage_context,)

#Re-ranking
rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)

query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[rerank] )

#Interaction
resp_txt = "How to take backup of oracle DB?" # @param {"type":"string"}
now = time.time()
response = query_engine.query(resp_txt,)
print(f"Response Generated: {response}")
print(f"Elapsed: {round(time.time() - now, 2)}s")
