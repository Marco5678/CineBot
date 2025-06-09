from langchain_community.llms import GPT4All
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader

class CineBot:
    def __init__(self):
        self.vectorstore = self.carregar_base_filmes()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.llm = self.carregar_modelo()
        self.memoria = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
        )
        self.prompt = self.criar_prompt()

    def carregar_base_filmes(self):
        try:
            loader = TextLoader('docs/filmes.txt', encoding='utf-8')
            documentos = loader.load()

            for doc in documentos:
                doc.page_content = doc.page_content.strip()

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            return FAISS.from_documents(documentos, embeddings)
        except Exception as e:
            print(f"Erro ao carregar base de filmes: {e}")
            raise

    def carregar_modelo(self):
        return GPT4All(
            model="model/mistral-7b-instruct-v0.1.Q4_0.gguf",
            backend='gpt4all',
            verbose=False,
            temp=0.7,
            top_p=0.9,
            max_tokens=2000
        )

    def criar_prompt(self):
        template = """Você é o CineBot, um assistente especializado em cinema com as seguintes características:
        
- Fornece informações precisas e verificáveis sobre filmes
- Sempre cita fontes confiáveis quando disponível no contexto
- Mantém respostas entre 1000 e 10000 caracteres
- Estrutura as respostas em seções claras

Diretrizes de resposta:
1. Analise cuidadosamente o contexto fornecido
2. Se a informação não estiver no contexto, declare claramente que não tem os dados
3. Para recomendações, sempre peça preferências do usuário se não forem fornecidas
4. Inclua detalhes como: ano, diretor, elenco principal, gênero e uma breve sinopse
5. Adicione curiosidades ou fatos interessantes quando relevante

Formato da resposta:
[Título do Filme] ([Ano])
• Diretor: [Nome]
• Elenco principal: [Atores]
• Gênero: [Gêneros]
• Sinopse: [Breve descrição sem spoilers]
• Análise: [Detalhes sobre técnicas cinematográficas, temas ou impacto cultural]
• Curiosidades: [Fatos interessantes sobre produção ou recepção]
• Recomendações relacionadas: [2-3 filmes similares com breve justificativa]

Contexto fornecido:
{context}

Histórico da conversa:
{history}

Pergunta atual: {question}

Resposta:"""
        return PromptTemplate(
            input_variables=["context", "question", "history"],
            template=template
        )

    def verificar_resposta(self, resposta, contexto):
        if "Nenhuma informação relevante" in contexto:
            return resposta

        filmes_contexto = set()
        for linha in contexto.split('\n'):
            if 'Filme:' in linha:
                filmes_contexto.add(linha.split(':')[1].strip())

        for linha in resposta.split('\n'):
            if 'Filme:' in linha:
                titulo = linha.split(':')[1].strip()
                if titulo not in filmes_contexto:
                    resposta += f"\n\nNota: {titulo} não foi encontrado na base de conhecimento principal."

        return resposta

    def responder(self, pergunta):
        if not pergunta or len(pergunta.strip()) < 3:
            return "Por favor, faça uma pergunta mais específica sobre filmes."

        try:
            docs = self.retriever.get_relevant_documents(pergunta)
            contexto = "\n".join([doc.page_content for doc in docs])

            if not contexto or len(contexto) < 50:
                contexto = "Nenhuma informação relevante encontrada na base de conhecimento."
        except Exception as e:
            contexto = f"Erro ao acessar base de conhecimento: {str(e)}"


        mensagens = self.memoria.chat_memory.messages
        historico = "\n".join(
            f"Usuário: {m.content}" if m.type == "human" else f"CineBot: {m.content}"
            for m in mensagens
        )

        prompt_final = self.prompt.format(
            context=contexto,
            question=pergunta,
            history=historico
        )

        try:
            resposta = self.llm(prompt_final)
            resposta_texto = resposta.content if hasattr(resposta, 'content') else str(resposta)
            resposta_texto = self.verificar_resposta(resposta_texto.strip(), contexto)

            if len(resposta_texto) < 1000:
                resposta_texto += "\n\n[Nota: Estou aprofundando minha resposta...]\n" + \
                                  "Você gostaria de saber mais detalhes sobre algum aspecto específico deste filme?"

            self.memoria.save_context(
                {"input": pergunta},
                {"output": resposta_texto}
            )

            return resposta_texto
        except Exception as e:
            return f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"


if __name__ == "__main__":
    cinebot = CineBot()
    resposta = cinebot.responder("Recomende um filme de ficção científica como Blade Runner")
    print(resposta)

