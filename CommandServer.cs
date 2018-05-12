using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;

public class CommandServer : MonoBehaviour
{
	public CarRemoteControl CarRemoteControl;
	public Camera FrontFacingCamera;
	private SocketIOComponent _socket;
	private CarController _carController;
	private bool frameUpdated = false;
	private bool physicsUpdated = false;
	private UInt64 frames= 0;
	private UInt64 pframes = 0;

	// Use this for initialization
	void Start()
	{
		_socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
		_socket.On("open", OnOpen);
		_socket.On("steer", OnSteer);
		_socket.On("manual", onManual);
		_carController = CarRemoteControl.GetComponent<CarController>();
	}

	// Update is called once per frame
	void Update()
	{
		frames++;
		frameUpdated = true;
	}

	void FixedUpdate()
	{
		pframes++;
		physicsUpdated = true;
	}

	void OnOpen(SocketIOEvent obj)
	{
		Debug.Log("Connection Open");
		//Debug.Log("Connection Open");
		//EmitTelemetry(obj);
	}
    	
	void onManual(SocketIOEvent obj)
	{
		EmitTelemetry (obj);
	}

	void OnSteer(SocketIOEvent obj)
	{
		Debug.Log ("on steer mesage being received...");
		JSONObject jsonObject = obj.data;
		//    print(float.Parse(jsonObject.GetField("steering_angle").str));
		CarRemoteControl.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
		EmitTelemetry(obj);
	}

	void EmitTelemetry(SocketIOEvent obj)
	{
		UnityMainThreadDispatcher.Instance().Enqueue(() =>
		{
			print("Attempting to Send...");
			// send only if it's not being manually driven
			if ((Input.GetKey(KeyCode.W)) || (Input.GetKey(KeyCode.S))) {
				_socket.Emit("telemetry", new JSONObject());
			}
			else {
				// Collect Data from the Car
				Dictionary<string, string> data = new Dictionary<string, string>();
				data["image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));

				physicsUpdated = false;
				frameUpdated = false;

				_socket.Emit("telemetry", new JSONObject(data));
			}
		});
	}
}