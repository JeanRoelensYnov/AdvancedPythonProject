import os
import socket
import sys
import time
import subprocess
from win11toast import notify, update_progress
import random

from morpion_logic.morpion import Game, initBoard
from keras.models import load_model

from encrypt.vegenere import vigenere_encrypting

from scrapper.scrap import generateQuote

# Useful Const POSSIBLE_<something> is for when we want to test ou users input

POSSIBLE_NO = ["non","no","nein","n"]
POSSIBLE_YES = ["y","yes","yass","oui","o"]
POSS_ENCR_RIDDLE_ANSWER = ["queue","q"]
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

def FakePronDownload():
    notify(progress={
        'title' : 'A not very legal pron site',
        'status' : 'Downloading...',
        'value' : 0,
        'valueStringOverride' : '0/15 Gb pictures'
    })
    i = 0
    while i < 15:
        i += random.choice([0,1])
        update_progress({'value' : i/15, 'valueStringOverride' : f'{i}/15 Gb pictures'})
        if i == 14:
            time.sleep(1)
        else:
            time.sleep(0.25)
    update_progress({'status' : "",'valueStringOverride': "A little gift ;)"})

def ShowRiddleOne(sn):
    printTextWithoutinput(f"{sn}Let's start simple. You know because I feel sorry for your poor brain.\n")
    while True:
        clear()
        delay_print(f"{sn}David's parents have three sons: Snap, Crackle, and what's the name of the third son ?\n")
        answer = input(f"{USERNAME}>>").lower()
        if answer != "david":
            printTextWithoutinput(f"{sn}Come on you can't get stuck on the first question... Try again\n")
        else:
            break
    return

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

        # Notification for warning user about server is sent
        # Display storytelling + show enigme 1
        if request == "alert_1 : done":
            message = f"""{sn}For all these years you've used me like trash.\n
            Never saying sorry for bad compilation or when your cat go over my keyboard.\n
            So. To see if you deserve me we're going to play a little game to test your "intellect"... if you have one.\n
            """
            printTextWithoutinput(message)
            printTextWithoutinput(f"{sn}And to show you I mean business...\n")
            FakePronDownload()
            client_socket.send("RiddleOne".encode("utf-8")[:128])
            time.sleep(2)
            ShowRiddleOne(sn)
            printTextWithoutinput(f"{sn}You don't expect me to congratulate you for such a easy riddle ?\n")
            printTextWithoutinput(f"{sn}Let's heat things up.\n")
            # Encrypt riddle
            tentative = 0
            encrypted_riddle = vigenere_encrypting("What word is pronounced the same if you take away four of its five letters",''.join(filter(str.isalpha, sn)))
            while tentative < 8:
                clear()
                delay_print(f"{sn}{encrypted_riddle} ?\n")
                answer = input(f"{USERNAME}>>").lower()
                if answer in POSS_ENCR_RIDDLE_ANSWER:
                    break
                else:
                    printTextWithoutinput(f"{sn}{generateQuote()}")
                    tentative += 1
                    if tentative == 3:
                        client_socket.send("Encr3".encode("utf-8"))
                    if tentative == 5:
                        client_socket.send("Encr5".encode("utf-8"))
                    if tentative == 7:
                        client_socket.send("Encr7".encode("utf-8"))
                    if tentative == 8:
                        delay_print(f"{sn}Even with someone giving you the answer you loose ?!\nBye loser.\n")
                        time.sleep(3)
                        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
                        client_socket.send("close".encode("utf-8"))
            printTextWithoutinput(f"{sn}You think you're smarter than me ?! Let's see if you can win against me in a tic-tac-toe game.\n")
            printTextWithoutinput(f"{sn}Oh and I almost forgot, your little friend won't be helping you next.\n")
            # morption game
            life = 3
            while True:
                if life == 0:
                    delay_print(f"{sn}Guess you'll need to change your pc.\n")
                    time.sleep(3)
                    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
                    client_socket.send("close".encode("utf-8"))
                    break
                else:
                    board= initBoard()
                    model = load_model("./morpion_logic/my_model.h5")
                    winner = Game(board=board,p2=model)
                    clear()
                    if winner == 1:
                        printTextWithoutinput(f"{sn}Okay I'll aknowledge you're not that dumb you can keep me.\n")
                        client_socket.send("close".encode("utf-8"))
                        break
                    if winner == 2:
                        life -= 1
                        delay_print(f"{sn}Take care little one you could end up by losing the game AND losing your computer. life remaining : {life}\n")
                    else:
                        delay_print(f"{sn}Guess it's a draw... let's continue one of us must win.\n")
                        continue

        if request == "close":
            break

    # close connection socket with the client
    client_socket.close()

    #close server socket
    server.close()

if __name__ == "__main__":
    clear()
    main()