import enchant


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
    for k in range(26):
        decrypted_msg = decrypt(encrypted_msg, k)
        if is_sentence(decrypted_msg, language):
            return k, decrypted_msg
    return None, None


msg = "Secret message"
encrypted = encrypt(msg, 21)
language = "en_US"
key, word_match = find_best_word(encrypted, language)

print(
    f"""----------Final----------
Key found: {key}
Decrypted message: {word_match}
------------+------------""")

