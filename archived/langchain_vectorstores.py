import config
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


api_key = config.api_key


def main():
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-3.5-turbo",
        temperature=0
    )
    url = "https://infobae.com"
    loader = WebBaseLoader(url)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    query = "What does Kicillof wait for?"
    # Define the prompt for the LLM to generate a search query
    rephrase_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                """Given the above conversation, generate a search query to look up """
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
