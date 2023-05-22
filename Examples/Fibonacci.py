def helper(prev, curr):
    return curr, prev + curr


def fibonacci(n):
    fibonacci_list = [0, 1]

    if n <= 1:
        return fibonacci_list[:n + 1]

    prev, curr, count = 0, 1, 2

    while count < n + 1:
        prev, curr = helper(prev, curr)
        fibonacci_list.append(curr)
        ...
        count += 1

    return fibonacci_list


# Fibonacci number to calculate
fibonacci_number = 100

# Calculate the Fibonacci sequence
fibonacci_sequence = fibonacci(fibonacci_number)

# Print the Fibonacci sequence
print(f"Fibonacci sequence up to {fibonacci_number}:")
print(fibonacci_sequence)
