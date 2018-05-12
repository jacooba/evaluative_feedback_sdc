import tensorflow as tf
import numpy as np
import getopt
import sys
import os

import json
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template
from io import BytesIO

import time

import constants as c
from p_net import Model

sio = socketio.Server()
app = Flask(__name__)
session_id = None

running = False
state = None
action = None
model = None
ep_reward = 0

@sio.on('telemetry')
def telemetry(sid, data):
    global state, action, model, session_id, ep_reward

    state = data["image"]
    action = model.get_action(state)

    print("--------------")
    print('Action: ', action)

@sio.on('connect')
def connect(sid, environ):
    global running, session_id, model
    print("connect ", sid)
    running = True
    session_id = sid

@sio.on('disconnect')
def disconnect(sid):
    global running, session_id
    running = False
    session_id = None

def send_control(action):
    sio.emit("steering_angle", data={
        "steering_angle": action.__str__()
    }, skip_sid=True)


def main():
    global running, model

if __name__ == '__main__':
    print('Initializing model...')
    model = Model()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
