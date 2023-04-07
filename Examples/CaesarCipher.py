import enchant


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


def is_sentence(sentence, language):
    """
    Determines whether the input sentence is a proper English sentence or not.
    Returns a boolean value.
    """
    d = enchant.Dict(language)
    words = sentence.split()
    for word in words:
        if not d.check(word):
            return False
    return True


def find_best_word(encrypted_msg, language):
    """
    Finds the first word in the input list that is a proper English word.
    Returns None if no such word is found.
    """
    matches = dict()
    for rotation in range(26):
        decrypted_msg = decrypt(encrypted_msg, rotation)
        flag = is_sentence(decrypted_msg, language)
        matches[rotation] = decrypted_msg if flag else None

    for rotation, decrypted in matches.items():
        if decrypted:
            return rotation, decrypted


msg = "Secret message"
encrypted = encrypt(msg, 21)
lang = "en_US"
key, word_match = find_best_word(encrypted, lang)

print(
    f"""----------Final----------
Key found: {key}
Decrypted message: {word_match}
------------+------------""")

