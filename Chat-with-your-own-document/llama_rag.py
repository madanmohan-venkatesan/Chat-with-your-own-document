def llama_rag():
  #Streamlit
  import streamlit as st
  #For loading PDF
  import pathlib
  from llama_index.core import SimpleDirectoryReader
  import pickle as pkl
  
  #For embedding
  from llama_index.embeddings.fastembed import FastEmbedEmbedding
  from llama_index.core import Settings
  
  #For prompt template
  from llama_index.core import PromptTemplate
  
  #For downloading model
  from huggingface_hub import login
  from streamlit import secrets as st_secrets  
  #For storing embedded vectors in vector store
  import torch
  
  from transformers import AutoModelForCausalLM,AutoTokenizer
  from llama_index.llms.huggingface import HuggingFaceLLM
  
  from llama_index.core import VectorStoreIndex,get_response_synthesizer
  from llama_index.core.retrievers import VectorIndexRetriever
  from llama_index.core.query_engine import RetrieverQueryEngine

  from llama_index.core import StorageContext
  from llama_index.vector_stores.qdrant import QdrantVectorStore
  import qdrant_client
  
  #For re-ranking
  from llama_index.core.postprocessor import SentenceTransformerRerank
  #Paths
  CURRENT_PATH=pathlib.Path(".")
  DATA_PATH=pathlib.Path.joinpath(CURRENT_PATH,"Data")
  DATA_PATH.mkdir(exist_ok=True)
  ACCESS_TOKEN=st_secrets.HF_TOKEN


  st.title("Chat with your own document")
  files=st.sidebar.file_uploader("Upload a file",type=".pdf",accept_multiple_files=True)
  button=st.sidebar.button("Process documents")
  if "button" not in st.session_state:
    st.session_state.button=0
  if button:
    st.session_state.button=1
  if st.session_state.button==1 and len(files) == 0:
    st.info("Please upload files to continue")
  if st.session_state.button==1 and len(files)>0:
    file_name=None
    for single_file in files:
      file_name=single_file.name
      with open(DATA_PATH.joinpath(file_name),'wb') as f:
        f.write(single_file.getvalue())
  
    @st.cache_data
    def load_data():
      #Load the PDF Document
      st.info("Starting to load data")
      documents = SimpleDirectoryReader(DATA_PATH).load_data()
      st.info("Completed loading")
      return documents
    
    documents_=load_data()
      
    @st.cache_data
    def load_embedding_model():
      embed_model =FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
      st.info("Sucessfully configured embeddings")
      return embed_model

    embed_model_=load_embedding_model()
    Settings.embed_model = embed_model_
    Settings.chunk_size = 512
      
    
    #System prompt
    system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

    #Download model  
    @st.cache_resource
    def get_LLM(system_prompt,ACCESS_TOKEN):
      login(ACCESS_TOKEN)
      st.info("Sucessfully Logged in to HuggingFace")
      query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")
      tokenizer = AutoTokenizer.from_pretrained(
      "meta-llama/Llama-3.2-1B-Instruct"
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
      tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
      model_name="meta-llama/Llama-3.2-1B-Instruct",
      device_map="cuda",
      stopping_ids=stopping_ids,
      tokenizer_kwargs={"max_length": 4096},
      model_kwargs={"torch_dtype": torch.float16}
      )
      return llm
    llm_=get_LLM(system_prompt,ACCESS_TOKEN)
    st.info("Sucessfully Loaded LLM")
    
    @st.cache_resource
    def set_vecdb(_docs,_emdbs,_llm):
      Settings.llm = _llm
      Settings.chunk_size = 512
      Settings.embed_model = _emdbs
      #Store embeddings in vector store
      
      client = qdrant_client.QdrantClient(location=":memory:",
      )
      vector_store = QdrantVectorStore(client=client,collection_name="test")
      storage_context = StorageContext.from_defaults(vector_store=vector_store)
      index = VectorStoreIndex.from_documents(_docs,storage_context=storage_context,)
      st.info("Sucessfully configured Qdrant")
      return index
    index_=set_vecdb(documents_,embed_model_,llm_)  

    #Re-ranking
    
    @st.cache_resource
    def set_rerank():
      rerank = SentenceTransformerRerank( model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3)
      
      return rerank
    reranker=set_rerank()  
    @st.cache_resource
    def set_retriever(_index_):

      # configure retriever
      retriever = VectorIndexRetriever(
        index=_index_,
        similarity_top_k=20,
      )
      return retriever
    
    retriever_=set_retriever(index_)
    # configure response synthesizer
    @st.cache_resource
    def set_synthesizer():

      response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",streaming=True
      )
      return response_synthesizer
    resp_synthesizer=set_synthesizer()
    # assemble query engine
    @st.cache_resource
    def set_query_engine(_retriever_,_resp_synthesizer):
      query_engine = RetrieverQueryEngine(
        retriever=_retriever_,
        response_synthesizer=_resp_synthesizer,
        node_postprocessors=[reranker]
      )
      return query_engine
    query_eng=set_query_engine(retriever_,resp_synthesizer)

    # Initialize chat history
    if "messages" not in st.session_state:
      st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
      with st.chat_message(message["role"]):
        st.markdown(message["content"])

  # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)


    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_bot = query_eng.query(str(prompt),)

            full_response=st.write_stream(response_bot.response_gen)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
