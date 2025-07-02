from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
# Load environment variables
load_dotenv()

# Setting the environment variables
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['PPLX_API_KEY'] = os.getenv('PPLX_API_KEY')

# Initializing the LLM that will be used
llm = init_chat_model(
    model='llama-3.1-sonar-small-128k-online',
    model_provider='perplexity'
)

# initializing the embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-mpnet-base-v2'
)

# Initializing the vector store
vector_store = InMemoryVectorStore(
    embedding=embeddings
)

documents = []
# Loading the text file
path = './cleaned_texts/'
if not os.path.exists(path):
    raise FileNotFoundError(f"The directory {path} does not exist.")
for file in os.listdir(path):
    if file.endswith('.txt'):
        file_path = os.path.join(path,file)
        loader = TextLoader(file_path=file_path)
        documents.extend(loader.load())
    else:
        print(f"Skipping {file}, not a text file.")
        continue

# splitting the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# adding the chunks to the vector store
_ = vector_store.add_documents(chunks)

# defining a prompt for the LLM
prompt = hub.pull(
    "rlm/rag-prompt",
    api_url='https://api.smith.langchain.com'
)
# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()