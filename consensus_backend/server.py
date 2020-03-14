from flask import Flask, jsonify

from algorithm.Topology import *
from algorithm.NormalAlgo import NormalAlgo
from algorithm.NoiseAlgo import NoiseAlgo
from algorithm.CryptoAlgo import CryptoAlgo

app = Flask(__name__)

@app.route("/")
def hello_world():
    return jsonify({
        "message": "Hello World"
    }), 200

@app.route("/topology")
def topology():
    topologies = [Paper, Mesh, Ring, Star, FullyConnected, Line ,Tree]
    return jsonify([t.__name__ for t in topologies]),200

@app.route("/normal")
def normal():
    return jsonify({
        "message": "Hello World"
    }), 200

@app.route("/noise")
def noise():
    return jsonify({
        "message": "Hello World"
    }), 200

@app.route("/crypto")
def crypto():
    return jsonify({
        "message": "Hello World"
    }), 200



if __name__ == "__main__":
    app.run(debug=True)

