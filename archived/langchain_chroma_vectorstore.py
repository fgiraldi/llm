import config
import time
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from chromadb_class import ReviewVectorDB


# Connect to ChromaDB
start_time = time.time()
chroma_client = ReviewVectorDB(persist_directory="./chroma_db")
collection = chroma_client.client.get_collection(name="app_reviews")  # Update with your collection name

# Retrieve all stored embeddings and metadata
results = collection.get(include=["documents", "embeddings", "metadatas"])
documents = results["documents"]
embeddings = results["embeddings"]  # Use the retrieved embeddings
end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to load {len(documents)} items from a ChromaDB")

api_key = config.api_key
openai_embedding_fn = OpenAIEmbeddings(openai_api_key=api_key)


def main():
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )
    # Use the retrieved embeddings to build the FAISS vectorstore
    vectorstore = FAISS.from_texts(documents, openai_embedding_fn)
    query = "What do users complain the most about the app? If possibe, explain about the 2 top most common issues."
    # Define the prompt for the LLM to generate a search query
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                """Given the provided list of user reviews for an app, generate a search query to look up """
                """to get information relevant to the conversation"""
            )
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,
        vectorstore.as_retriever(),
        rephrase_prompt
    )
    chat_history = []

    history_aware_retriever.invoke({
        "chat_history": chat_history,
        "input": query
    })

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
    chat_history = []
    second_invoke = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    print(second_invoke["answer"])


if __name__ == "__main__":
    main()
