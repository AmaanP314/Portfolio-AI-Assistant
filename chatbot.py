import os
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict, Any, Optional
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.messages import  BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
embed_model = os.getenv("EMBEDDING_MODEL")
llm_model = os.getenv("LLM_MODEL")

embeddings = GoogleGenerativeAIEmbeddings(
    google_api_key=GOOGLE_API_KEY,
    model=embed_model
)
bm25_encoder = BM25Encoder().default()

index_name = "personal-assistant"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

class CustomHybridSearchRetriever(PineconeHybridSearchRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get documents relevant to the query using hybrid search with fallback to dense-only."""
        try:
            # Try hybrid search first
            return super()._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            # If sparse encoding fails, fall back to dense-only search
            if "Sparse vector must contain at least one value" in str(e):
                print("Falling back to dense-only search for query:", query)
                # Generate dense embeddings
                embedding = self.embeddings.embed_query(query)
                # Search with only dense vectors
                results = self.index.query(
                    vector=embedding,
                    top_k=self.top_k,
                    include_metadata=True,
                    namespace=self.namespace,
                )
                # Convert Pinecone results to LangChain documents
                return self._process_pinecone_results(results)
            else:
                # If it's a different error, re-raise it
                raise e
    
    def _process_pinecone_results(self, results):
        """Process Pinecone results into Document objects."""
        docs = []
        for result in results.matches:
            metadata = result.metadata or {}
            # Create Document with page content and metadata
            doc = Document(
                page_content=metadata.pop("text", ""),
                metadata=metadata,
            )
            docs.append(doc)
        return docs
    
retriever = CustomHybridSearchRetriever( 
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=3
    )

llm = ChatGoogleGenerativeAI(
    model=llm_model,
    google_api_key=GOOGLE_API_KEY,
    temperature=1.0,
)

store = {}

def get_full_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        print(f"INFO: Creating new chat history for session: {session_id}")
        store[session_id] = ChatMessageHistory()
    return store[session_id] 

MAX_HISTORY_TURNS = 3 
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2


def limit_history_for_rag_chain(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    modified_input = input_dict.copy()

    if "chat_history" in modified_input:
        history = modified_input["chat_history"]
        if isinstance(history, list) and all(isinstance(m, BaseMessage) for m in history):
            limited_history = history[-MAX_HISTORY_MESSAGES:]
            modified_input["chat_history"] = limited_history
        else:
             print("WARN: 'chat_history' in input_dict is not a list of BaseMessages. Passing as is.")
    return modified_input

retriever_prompt_template = os.getenv("RETRIEVER_PROMPT").format(max_turns=MAX_HISTORY_TURNS)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", retriever_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"), # This will receive the limited history
        ("human", "{input}"),
     ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT").format(max_turns=MAX_HISTORY_TURNS)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

conversational_rag_chain = RunnableWithMessageHistory(
    runnable=RunnableLambda(limit_history_for_rag_chain) | rag_chain, 
    get_session_history=get_full_session_history, 
    input_messages_key="input",
    history_messages_key="chat_history", 
    output_messages_key="answer",
)
def chat(query: str, session_id: str):   
    response = conversational_rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )
    return response.get("answer", "No answer found.")