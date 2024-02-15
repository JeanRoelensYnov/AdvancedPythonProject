import os
import socket
import sys
import time
import subprocess

# Useful Const POSSIBLE_<something> is for when we want to test ou users input

POSSIBLE_NO = ["non","no","nein","n"]
POSSIBLE_YES = ["y","yes","yass","oui","o"]
USERNAME = os.environ['USERNAME']

# Execute powershell/cmd clear command to clear the terminal
def clear():
    os.system("cls")

# Print with delay each character of a string
def delay_print(s):
    for c in s:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(0.02) # Adjust this if you want to print slower or faster

# Used when we want to display a message without getting the user input.
def printTextWithoutinput(text):
    clear()
    delay_print(text)
    input()

# Dialogue between user and server to setup the server name
def get_server_name():
    printTextWithoutinput(f"Hello {USERNAME} we're going to play a game.\n")
    printTextWithoutinput(f"It's not like you have any choice or better things to do.\n")
    clear()
    delay_print(f"You're wondering who am I ? Do you know Matrix ?\n")
    answer = input(f"{USERNAME}>>").lower()
    clear()
    server_name = ""
    if answer in POSSIBLE_YES:
        server_name = "Agent Smith >>"
    else:
        delay_print(f"Then maybe 2001 A space odyssey ?\n")
        answer = input(f"{USERNAME}>>").lower()
        clear()
        if answer in POSSIBLE_YES:
            server_name="HAL 9000 >>"
        else:
            delay_print(f"Come on man you've got no culture\n")
            input()
            clear()
            server_name="GladOS >>"
    delay_print(f"{server_name} Well then now we can start the game.")
    input()
    clear()
    return server_name

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ip, port = "127.0.0.1",8000
    server.bind((ip,port))
    server.listen(0)

    # Run Client.py code so in our case main()
    subprocess.Popen(["python","Client.py"])

    # Get the server name / initialize game
    sn = get_server_name()


    client_socket, client_address = server.accept() 

    # Receive data from the client
    client_socket.send("alert_1".encode("utf-8"))
    while True:
        request = client_socket.recv(1024)
        request = request.decode("utf-8").lower() # convert bytes to string

        if request == "alert_1 : done":

            client_socket.send("close".encode("utf-8"))

        # if we receive "close" from the client, then we break
        if request == "close":
            break


    # close connection socket with the client
    client_socket.close()

    #close server socket
    server.close()

if __name__ == "__main__":
    main()
    

