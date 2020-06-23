from socket import socket

HOST = ''
PORT = 12345

sock = socket()
sock.bind((HOST, PORT))

sock.listen(1)
conn = sock.accept()

data = conn.recv(2048).decode()