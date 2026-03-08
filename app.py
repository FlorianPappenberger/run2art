from flask import Flask, jsonify, send_from_directory
import os
import json

app = Flask(__name__, static_folder="public")

@app.route("/api/results")
def get_results():
    results_path = os.path.join(app.static_folder, "benchmark_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Results not found"}), 404

@app.route("/api/comparison")
def get_comparison():
    comparison_path = os.path.join(app.static_folder, "engine_comparison.json")
    if os.path.exists(comparison_path):
        with open(comparison_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Comparison not found"}), 404

@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(debug=True)