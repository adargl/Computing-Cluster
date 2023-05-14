from random import randint


def quick_sort(array):
    # Initialize a stack with the array and its indices
    stack = [(0, len(array)-1)]

    # While the stack is not empty
    while stack:
        # Pop the indices of the current subarray
        start, end = stack.pop()
        # If the subarray has more than one element, partition it
        if start < end:
            # Partition the array and get the index of the pivot element
            pivot = partition(array, start, end)
            # Push the subarrays to the left and right of the pivot element to the stack
            stack.append((start, pivot-1))
            stack.append((pivot+1, end))
            # Sort the subarrays
            sort_subarray(array, start, pivot-1)
            sort_subarray(array, pivot+1, end)

    return array


def partition(array, start, end):
    # Select the pivot element
    pivot = array[end]
    # Initialize the pivot index
    i = start - 1
    # Iterate over the subarray
    for j in range(start, end):
        # If the current element is smaller than the pivot element, swap it with the pivot index
        if array[j] < pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    # Swap the pivot element with the pivot index
    array[i+1], array[end] = array[end], array[i+1]
    # Return the pivot index
    return i + 1


def sort_subarray(array, start, end):
    # If the subarray has more than one element, sort it using insertion sort
    for i in range(start+1, end+1):
        j = i
        while j > start and array[j] < array[j-1]:
            array[j], array[j-1] = array[j-1], array[j]
            j -= 1


# Test the quick_sort function
lst = list(map(lambda x: randint(1, 100), range(50)))
print(quick_sort(lst))
