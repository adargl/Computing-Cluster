import hashlib


def get_hash(string):
    return hashlib.sha256(string.encode()).hexdigest()


def find_nonce(block_header, difficulty):
    # Combine the block header and nonce into a single string
    input_str = str(block_header)

    # Start with a nonce of 0
    nonce = 0

    while True:
        # Add the nonce to the input string
        input_str_with_nonce = input_str + str(nonce)

        # Calculate the SHA-256 hash of the input string with the nonce
        hash_result = get_hash(input_str_with_nonce)

        # Check if the hash meets the difficulty requirement
        if hash_result.startswith('0' * difficulty):
            return nonce, hash_result

        nonce += 1


# Sample block header
block_header = {
    "previous_block_hash": "0000000000000000000000000000000000000000000000000000000000000000",
    "transactions": ["transaction_1", "transaction_2", "transaction_3"],
    "timestamp": "2022-01-01 12:00:00"
}

# Difficulty level
difficulty = 4

# Find nonce and hash value
nonce, hash_value = find_nonce(block_header, difficulty)

# Print results
print("Nonce found: {}".format(nonce))
print("Hash value: {}".format(hash_value))
