from flask import Flask, render_template, request, jsonify
from chatbot import CineBot

app = Flask(__name__)
chatbot = CineBot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    pergunta = data.get("message")
    resposta = chatbot.responder(pergunta)
    return jsonify({"response": resposta})

if __name__ == "__main__":
    app.run(debug=True)