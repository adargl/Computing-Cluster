import enchant.checker as chkr
import time

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


def is_sentence(sentence, checker):
    """
    Determines whether the input sentence is a proper English sentence or not.
    Returns a boolean value.
    """
    time.sleep(0.2)
    checker.set_text(sentence)
    for _ in checker:
        return False
    return True


def find_best_word(encrypted_msg, checker):
    """
    Finds the first word in the input list that is a proper English word.
    Returns None if no such word is found.
    """
    matches = dict()
    for rotation in range(26):
        decrypted_msg = decrypt(encrypted_msg, rotation)
        flag = is_sentence(decrypted_msg, checker)
        matches[rotation] = decrypted_msg if flag else None

    for rotation, decrypted in matches.items():
        if decrypted:
            return rotation, decrypted


# Initialize enchant dictionary
language = "en_US"
spell_checker = chkr.SpellChecker(language)

msg = "Secret message"
encrypted = encrypt(msg, 21)
key, word_match = find_best_word(encrypted, spell_checker)

print(f"""Key found: {key}
Decrypted message: {word_match}""")

