def vigenere_encrypting(mess, key):
    result = ""
    whitespace_index = [i for i, char in enumerate(mess) if char == " "]
    mess, key = mess.lower().replace(" ", ""), key.lower()
    # duplicate the key to match length of message
    repeated_key = key * (len(mess)//len(key)) + key[:len(mess)]
    #encrypt
    for i in range(len(mess)):
        if mess[i].isalpha():
            char = ord(mess[i]) + ord(repeated_key[i]) - 97
            char = char - 26 if char > 122 else char
        else:
            char = ord(mess[i])
        result += chr(char)

    for i in range(len(whitespace_index)):
        result = result[:whitespace_index[i]] + " " + result[whitespace_index[i]:]
    return result
