from langchain_classic.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore


def get_ai_message(user_message):

    # embedding model
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    index_name = "tax-markdown-index"

    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)

    llm = ChatOllama(model="exaone3.5:7.8b")

    # custom Korean prompt for tax QA
    prompt = ChatPromptTemplate.from_template(
        """다음 context를 기반으로 질문에 답변해주세요.
        context에 없는 내용은 모른다고 답변해주세요.

        Context: {context}

        Question: {question}"""
    )

    # retrieve document based on query
    retriever = database.as_retriever(search_kwargs={"k": 5})

    # define QA Chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})

    # define dictionary
    dictionary = ["사람을 나타내는 표현 -> 거주자"]

    dictionary_prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서, 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        변경된 질문만 출력해주세요. 다른 설명은 하지 마세요.

        사전: {dictionary}

        질문: {{question}}
    """
    )

    dictionary_chain = dictionary_prompt | llm | StrOutputParser()

    # build tax_chain (dictionary_chain + qa_chain)
    tax_chain = {"query": dictionary_chain} | qa_chain
    ai_message = tax_chain.invoke({"question": user_message})

    return ai_message["result"]
