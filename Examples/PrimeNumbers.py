import time


def is_prime(number):
    # Simulate a calculation
    time.sleep(0.1)

    # Check if the number is 1 or less
    if number <= 1:
        return False

    # Check if the number is divisible by any number from 2 to its square root
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False

    # If the number is not divisible by any number from 2 to its square root, it is prime
    return True


highest_number = 50
primes = [0 for _ in range(highest_number)]
for num in range(1, highest_number):
    result = is_prime(num)
    if result:
        primes[num] = num

print([prime for prime in primes if prime])