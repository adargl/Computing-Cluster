import time
import nltk
from nltk.corpus import words


def encrypt(message, rotation):
    result = ""
    # transverse the plain text
    for i in range(len(message)):
        char = message[i]
        if not char.isalpha():
            result += char
        # Encrypt uppercase characters in plain text
        elif char.isupper():
            result += chr((ord(char) + rotation - 65) % 26 + 65)
        # Encrypt lowercase characters in plain text
        else:
            result += chr((ord(char) + rotation - 97) % 26 + 97)
    return result


def decrypt(message, rotation):
    result = ""
    for char in message:
        if char.isalpha():
            if char.isupper():
                result += chr((ord(char) - rotation - 65) % 26 + 65)
            else:
                result += chr((ord(char) - rotation - 97) % 26 + 97)
        else:
            result += char
    return result


def is_word(word):
    word_set = set(words.words())
    return word.lower() in word_set


def is_sentence(sentence):
    all_words = sentence.split()
    for word in all_words:
        if not is_word(word):
            return False
    return True


def find_best_match(encrypted_msg):
    matches = dict()
    for rotation in range(26):
        time.sleep(0.1)
        decrypted_msg = decrypt(encrypted_msg, rotation)
        flag = is_sentence(decrypted_msg)
        matches[rotation] = decrypted_msg if flag else None

    for rotation, decrypted in matches.items():
        if decrypted:
            return rotation, decrypted


# Download the English word corpus
nltk.download('words')

msg = "Secret message"
encrypted = encrypt(msg, 21)
key, word_match = find_best_match(encrypted)

print(f"""Key found: {key}
Decrypted message: {word_match}""")

