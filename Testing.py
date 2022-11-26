import ast
import multiprocessing


def encrypt(message, key):
    result = ""
    # transverse the plain text
    for i in range(len(message)):
        char = message[i]
        if not char.isalpha():
            result += char
        # Encrypt uppercase characters in plain text
        elif char.isupper():
            result += chr((ord(char) + key - 65) % 26 + 65)
        # Encrypt lowercase characters in plain text
        else:
            result += chr((ord(char) + key - 97) % 26 + 97)
    return result


def decrypt(message, key):
    result = ""
    for char in message:
        if char.isalpha():
            if char.isupper():
                result += chr((ord(char) - key - 65) % 26 + 65)
            else:
                result += chr((ord(char) - key - 97) % 26 + 97)
        else:
            result += char
    return result


def container(msg, encrypted):
    for key in range(26):
        if decrypt(encrypted, key) == msg:
            global k
            k = key

msg = "Secret message"
k = 21
encrypted = encrypt(msg, k)
print("Encrypted message:", encrypted)
k = 10

container(msg, encrypted)

print(
    f"""----------Final----------
Key found: {k}
Decrypted message: {decrypt(encrypted, k)}
------------+------------""")

# print("Key found:", k)
# print("Decrypted message:", decrypt(encrypted, k))
