from random import randint


def merge_sort(array):
    global a, s, t
    array = array.copy()
    subarrays = [[x] for x in array]

    # While there is more than one subarray, merge pairs of adjacent subarrays
    while len(subarrays) > 1:
        merged = []
        for i in range(0, len(subarrays), 2):
            # Merge the subarrays if there are two
            if i + 1 < len(subarrays):
                merged.append(merge(subarrays[i], subarrays[i + 1]))
            # Append the leftover subarray
            else:
                merged.append(subarrays[i])
        subarrays = merged

    # Return the only remaining subarray
    return subarrays[0]


def merge(left, right):
    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        # If the element from the left subarray is smaller, append it to the merged list
        if left[i] < right[j]:
            merged.append(left[i])
            i += 1
        # Otherwise, append the element from the right subarray
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


# Test the merge_sort function
lst = list(map(lambda x: randint(1, 100), range(100)))
print(merge_sort(lst))
