import os

from chromadb.errors import NoIndexException
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "db")


def create_prompt():
    prompt_template = """Use the following pieces of context to answer the question at the end.
     If you don't know the answer, just answer with "I don't know", don't try to make up an answer.

    {context}

    Question: {question}
    Answer in English:"""
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )


def get_documents():
    urls = ["https://www.w3.org/TR/WCAG21/"]
    return WebBaseLoader(urls).load()


def create_vector_db(collection_name, embeddings):
    document_id_prefix = "wcag"

    documents = get_documents()
    text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    collections = {}

    for i in range(len(docs)):
        collections[f"{document_id_prefix}-{i}"] = docs[i]

    vector_db = Chroma.from_documents(
        collection_name=collection_name,
        documents=list(collections.values()),
        embedding=embeddings,
        ids=list(collections.keys()),
        persist_directory=DB_DIR
    )

    return vector_db


def get_vector_search(collection_name):
    embeddings = HuggingFaceEmbeddings()
    vector_db = Chroma(
        collection_name=collection_name,
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    try:
        if vector_db._collection.count() == 0:
            raise NoIndexException
    except NoIndexException:
        vector_db = create_vector_db(collection_name, embeddings)
    finally:
        return vector_db


def create_compressed_retrival(hub_llm, collection_name):
    vector_search = get_vector_search(collection_name)
    retriever = vector_search.as_retriever(search_kwargs={"k": 4, "type": "mmr"})
    compressor = LLMChainExtractor.from_llm(hub_llm)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)


def query_pages():
    repo_id = "google/flan-t5-base"
    vector_db_collection_name = "wcag_collection"
    model_config = {"temperature": 0, "max_length": 50}
    qa_chain_type = "stuff"
    questions_to_ask = [
        "Who is creator of WCAG?",
        "What is 'Text Alternatives'?",
        "What does WCAG stand for?",
    ]

    hub_llm = HuggingFaceHub(repo_id=repo_id, model_kwargs=model_config)
    qa_chain = load_qa_chain(hub_llm, chain_type=qa_chain_type, prompt=create_prompt(), verbose=True)
    qa = RetrievalQA(
        combine_documents_chain=qa_chain,
        retriever=create_compressed_retrival(hub_llm, vector_db_collection_name)
    )

    for query in questions_to_ask:
        print(f"Question: {query}")
        print(f"Answer: {qa.run(query)}\n")


if __name__ == "__main__":
    query_pages()
