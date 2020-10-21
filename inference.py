import socket

def run_inf(port=5559, host='127.0.0.1', timeout=1000):
	"""
	Function to run inference on the local server where the pretrained model is running.
	:param port: The port through which the communication with the model happens.
	:param host: The local address of the server.
	:param timeout: A specified amount of time to wait for a response from the server before closing.
	"""
	# Connect to server
	#print("Connecting to server...")

	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.connect((host,port))

		s.sendall(b'run model!')

		#send message to server
		#print(f'Sending request...')

		data = s.recv(1024).decode()
		print (data)