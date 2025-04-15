from flask import Flask, render_template, request, jsonify
import random
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import unicodedata

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

filmes = [
    {"titulo": "John Wick", "genero": "ação"}, {"titulo": "Missão Impossível", "genero": "ação"}, {"titulo": "Resgate", "genero": "ação"}, {"titulo": "Gladiador", "genero": "ação"},
    {"titulo": "As Branquelas", "genero": "comédia"}, {"titulo": "Minha Mãe é uma Peça", "genero": "comédia"}, {"titulo": "Click", "genero": "comédia"}, {"titulo": "Ace Ventura", "genero": "comédia"},
    {"titulo": "Invocação do Mal", "genero": "terror"}, {"titulo": "It", "genero": "terror"}, {"titulo": "Corra", "genero": "terror"}, {"titulo": "A Morte Te Dá Parabéns", "genero": "terror"},
    {"titulo": "À Procura da Felicidade", "genero": "drama"}, {"titulo": "Clube da Luta", "genero": "drama"}, {"titulo": "Forrest Gump", "genero": "drama"}, {"titulo": "Os Intocáveis", "genero": "drama"},
    {"titulo": "Simplesmente Acontece", "genero": "romance"}, {"titulo": "Como Eu Era Antes de Você", "genero": "romance"}, {"titulo": "Questão de Tempo", "genero": "romance"}, {"titulo": "Diário de uma Paixão", "genero": "romance"},
    {"titulo": "Divertida Mente", "genero": "animação"}, {"titulo": "Encanto", "genero": "animação"}, {"titulo": "Moana", "genero": "animação"}, {"titulo": "Shrek", "genero": "animação"},
    {"titulo": "Interestelar", "genero": "ficção"}, {"titulo": "A Origem", "genero": "ficção"}, {"titulo": "Matrix", "genero": "ficção"}, {"titulo": "Gravidade", "genero": "ficção"},
]

chaves_genero = {
    "ação": ["ação", "aventura", "explosão"],
    "comédia": ["comédia", "engraçado", "rir", "humor"],
    "terror": ["terror", "assustador", "medo", "horror"],
    "drama": ["drama", "emocionante"],
    "romance": ["romance", "amor", "relacionamento"],
    "animação": ["animação", "desenho", "infantil"],
    "ficção": ["ficção", "científica", "espaço"]
}

contexto_usuario = {
    "genero_atual": None,
    "sugeridos": []
}

mensagens_saudacao = [
    "E aí! Tá a fim de uma indicação de filme? Me diz um gênero.",
    "Olá! Me fala um estilo de filme que você curte pra eu te sugerir algo."
]

mensagens_sugestao = [
    "Você pode gostar de '{filme}'! É um ótimo exemplo de filme de {genero}. Quer outra indicação?",
    "Já viu '{filme}'? É um dos melhores de {genero}! Quer outra indicação?",
    "Minha sugestão é '{filme}', um excelente filme de {genero}! Quer outra indicação?",
    "Que tal assistir '{filme}'? Uma bela escolha de {genero}. Quer outra indicação?",
]

mensagens_despedida = [
    "Tchau! Até a próxima!",
    "Valeu pela conversa! Qualquer hora posso sugerir mais filmes.",
    "Até mais! Quando quiser, é só chamar.",
    "Foi um prazer ajudar! Até logo!"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    mensagem = request.get_json().get("message", "").lower()
    resposta = responder_usuario(mensagem, contexto_usuario)
    return jsonify({"response": resposta})

def responder_usuario(mensagem, contexto):
    palavras_chave = extrair_palavras_chave(mensagem)

    if any(p in palavras_chave for p in ["tchau", "adeus", "falou", "flw", "fui", "até mais", "até logo", "valeu", "ate breve"]):
        contexto["genero_atual"] = None
        contexto["sugeridos"] = []
        return random.choice(mensagens_despedida)

    genero_detectado = detectar_genero(palavras_chave)
    if genero_detectado:
        contexto["genero_atual"] = genero_detectado
        contexto["sugeridos"] = []
        return sugerir_filme(genero_detectado, contexto)

    if any(p in palavras_chave for p in ["sim", "quero", "pode", "aceito", "claro", "concerteza","Gostaria"]):
        if contexto["genero_atual"]:
            return sugerir_filme(contexto["genero_atual"], contexto)
        else:
            return "Antes me diga qual é o gênero de filme que você curte"

    if any(p in palavras_chave for p in ["oi", "ola", "eai", "e ai", "bom dia", "boa tarde", "boa noite"]):
        return random.choice(mensagens_saudacao)

    if any(p in palavras_chave for p in ["nao"]): 
        return "Tranquilo! Quando quiser mais indicações de filmes, é só me dizer o gênero."

    return "Não entendi muito bem. Me diz um gênero de filme e eu te sugiro algo legal!"

def sugerir_filme(genero, contexto):
    opcoes = [f["titulo"] for f in filmes if f["genero"] == genero and f["titulo"] not in contexto["sugeridos"]]
    if opcoes:
        filme = random.choice(opcoes)
        contexto["sugeridos"].append(filme)
        resposta = random.choice(mensagens_sugestao).format(filme=filme, genero=genero)
        return resposta
    else:
        return f"Já falei todos os filmes de {genero} que conheço. Que tal tentar outro gênero?"

def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def extrair_palavras_chave(texto):
    texto = remover_acentos(texto.lower())
    tokenizer = RegexpTokenizer(r'\w+')
    palavras = tokenizer.tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    return [p for p in palavras if p not in stop_words]

def detectar_genero(palavras_chave):
    for genero, chaves in chaves_genero.items():
        if any(remover_acentos(p) in [remover_acentos(c) for c in chaves] for p in palavras_chave):
            return genero
    return None

if __name__ == "__main__":
    print("Servidor Flask iniciado em http://127.0.0.1:5000")
    app.run(debug=True)
