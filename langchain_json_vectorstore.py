import config
import json
import time
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
# from chromadb_class import ReviewVectorDB


# Connect to ChromaDB
start_time = time.time()

json_path = "reviews_with_embeddings.json"

# Retrieve all stored embeddings and metadata
# 1. Load your JSON file with documents and embeddings
with open(json_path, "r") as f:
    data = json.load(f)

# 2. Prepare documents and embeddings
# Assuming your JSON structure has documents and their embeddings
documents = []
embeddings_list = []
metadatas = []

# Parse your JSON structure - adjust based on your actual JSON format
for item in data:
    # Extract text content
    doc_text = item["review"]  # or however your document text is stored

    # Extract embedding vector
    embedding = item["embedding"]  # Assuming this is already a list of floats

    # Extract metadata if available
    metadata = {
        "category": item.get("category", ""),
        "date": item.get("date", ""),
    }

    documents.append(doc_text)
    embeddings_list.append(embedding)
    metadatas.append(metadata)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"It took {elapsed_time} seconds to load {len(documents)} items from {json_path}")

api_key = config.api_key
openai_embedding_fn = OpenAIEmbeddings(openai_api_key=api_key)


def main():
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )
    # Use the retrieved embeddings to build the FAISS vectorstore
    # vectorstore = FAISS.from_texts(documents, openai_embedding_fn)
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(documents, embeddings_list)),
        embedding=openai_embedding_fn,
        metadatas=metadatas if metadatas and all(metadatas) else None
    )
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
