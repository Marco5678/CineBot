from flask import Flask, render_template, request, jsonify
from chatbot import criar_chatbot

app = Flask(__name__)
chatbot = criar_chatbot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    pergunta = data.get("message")
    resposta = chatbot(pergunta)
    return jsonify({"response": resposta})

if __name__ == "__main__":
    app.run(debug=True)