from flask import Flask, request, jsonify
from rag_engine import RAGEngine

app = Flask(__name__)
engine = RAGEngine()

@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok"}), 200

@app.route("/ingest", methods=["POST"])
def ingest() -> tuple:
    data = request.get_json(force=True) or {}
    documents = data.get("documents", [])
    metadata = data.get("metadata", {})
    count = engine.ingest(documents, metadata)
    return jsonify({"ingested": count}), 200

@app.route("/query", methods=["POST"])
def query() -> tuple:
    data = request.get_json(force=True) or {}
    question = data.get("question", "")
    top_k = int(data.get("top_k", 3))
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400
    answer, sources = engine.query(question, top_k=top_k)
    return jsonify({"answer": answer, "sources": sources}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)