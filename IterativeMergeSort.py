from random import randint

def merge_sort(array):
    # Make a copy of the array to avoid modifying the original
    array = array[:]

    # Start with a list of one-element lists
    subarrays = [[x] for x in array]

    # While there is more than one subarray, merge pairs of adjacent subarrays
    while len(subarrays) > 1:
        # Initialize the list of merged subarrays
        merged = []
        # Iterate over pairs of adjacent subarrays
        for i in range(0, len(subarrays), 2):
            # Merge the subarrays
            merged.append(merge(subarrays[i], subarrays[i + 1]))
        # Replace the list of subarrays with the list of merged subarrays
        subarrays = merged

    # Return the only remaining subarray
    return subarrays[0]


def merge(left, right):
    # Initialize the merged list
    merged = []
    # Initialize indices for the left and right subarrays
    i = j = 0
    # Iterate over the elements of the subarrays
    while i < len(left) and j < len(right):
        # If the element from the left subarray is smaller, append it to the merged list
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        # Otherwise, append the element from the right subarray
        else:
            merged.append(right[j])
            j += 1
    # Append the remaining elements of the left subarray, if any
    merged.extend(left[i:])
    # Append the remaining elements of the right subarray, if any
    merged.extend(right[j:])
    # Return the merged list
    return merged


# Test the merge_sort function
array = list(map(lambda x: randint(1, 100), range(100)))
print(merge_sort(array))
