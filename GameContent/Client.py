from os import path
import os
import socket
from winotify import Notification
import random


# Get username
def getUsername():
    """
    Since our script is polite it should address to our user by his name
    (Okay we do that just to add a sark of creepyness/edgyness)
    """
    return os.environ["USERNAME"]

# To get correctly basic Notification object
def NotificationBuilder(m):
    """
    Since we will be using notification to interact with our user and we don't
    want to rewrite Notification builder everytime we just pass it through a function
    """
    return Notification(
        app_id= "A friend who wish you the best",
        title= getSha(),
        msg= m,
        duration= "short",
    )

# Random bit generator generator
def getSha():
    return random.getrandbits(64)
    
def main():
    #Create a socket object
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = "127.0.0.1"
    server_port = 8000
    client.connect((server_ip,server_port))
    while True:
        # Input message and send it to the server

        # Receive message from the server
        response = client.recv(1024)
        response = response.decode("utf-8").lower()

        # If server sent us "closed" in the payload, we break out of the loop and close our socket
        if response == "alert_1":
            a = NotificationBuilder("Don't listen to him !\nHe's gonna break your computer")
            a.show()
            client.send("alert_1 : done".encode("utf-8")[:1024])
        if response == "riddleone":
            a = NotificationBuilder("Okay just play along with him I'll try to help you.")
            a.show()
        if response == "encr3":
            a = NotificationBuilder("Hey it looks like vigenere encryption !")
            a.show()
        if response == "encr5":
            a = NotificationBuilder("The key must be something that you see since the beginning...\nSomething that's here right before you eye.")
            a.show()
        if response == "encr7":
            a = NotificationBuilder("I got it ! By decrypting the riddle I found the answer 'Queue'")
            a.show()

        if response == "close":
            client.send("close".encode("utf-8")[:1024])
            break

        # print(f"Received : {response}")
    client.close()

main()
