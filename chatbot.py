from langchain_community.llms import GPT4All
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader


def carregar_base_filmes():
    loader = TextLoader('docs/filmes.txt')
    documentos = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documentos, embeddings)


def carregar_modelo():
    return GPT4All(
        model="model/mistral-7b-instruct-v0.1.Q4_0.gguf",
        backend='llama',  # Corrigido
        verbose=False
    )


def criar_prompt():
    template = """
    Você é um especialista em filmes. Use as informações abaixo (contexto) para responder à pergunta do usuário.

    Contexto:
    {context}

    Conversa até agora:
    {history}

    Pergunta:
    {question}
    """
    return PromptTemplate(
        input_variables=["context", "question", "history"],
        template=template
    )


def criar_chatbot():
    vectorstore = carregar_base_filmes()
    retriever = vectorstore.as_retriever()
    llm = carregar_modelo()
    memoria = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    prompt = criar_prompt()

    def responder(pergunta):

        docs = retriever.invoke(pergunta)
        contexto = "\n".join([doc.page_content for doc in docs])


        historico = memoria.load_memory_variables({})["history"]


        prompt_final = prompt.format(context=contexto, question=pergunta, history=historico)


        resposta = llm.invoke(prompt_final)


        resposta_texto = resposta.content if hasattr(resposta, 'content') else str(resposta)


        memoria.save_context({"input": pergunta}, {"output": resposta_texto})

        return resposta_texto

    return responder

