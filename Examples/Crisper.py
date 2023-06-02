import sys
import time


def compare(sequence1, sequence2, offset=0):
    # Simulate a calculation
    time.sleep(0.1)

    constants1 = ["NDEQ", "MILV", "FYW", "NEQK", "QHRK", "HY", "STA", "NHQK", "MILF"]
    constants2 = ["SAG", "SGND", "NEQHRK", "ATV", "STPA", "NDEEQHK", "HFY", "CSA", "STNK", "SNDEQK", "FVLIM"]
    longer_seq, shorter_seq = (sequence1, sequence2) if len(sequence1) > len(sequence2) else (sequence2, sequence1)
    longer_seq = longer_seq[offset:]

    new_seq = ""
    for i in range(len(shorter_seq)):
        if longer_seq[i] == shorter_seq[i]:
            new_seq += "*"
        else:
            for group in constants1:
                if longer_seq[i] in group and shorter_seq[i] in group:
                    new_seq += ":"
                    break
            else:
                for group in constants2:
                    if longer_seq[i] in group and shorter_seq[i] in group:
                        new_seq += "."
                        break
                else:
                    new_seq += " "
    return new_seq


def get_score(output):
    w1, w2, w3, w4 = 2, 1.5, 1.1, 1.3
    s = w1 * output.count("*") - w2 * output.count(":") - w3 * output.count(".") - w4 * output.count(" ")
    return s


def find_best_fit(sequence1, sequence2):
    offset = 0
    offset_to_score = dict()
    longer_seq, shorter_seq = (sequence1, sequence2) if len(sequence1) > len(sequence2) else (sequence2, sequence1)

    while len(longer_seq) > len(shorter_seq) + offset:
        seq = compare(sequence1, sequence2, offset)
        offset_to_score[offset] = get_score(seq)
        ...
        offset += 1

    max_pair = 0, -sys.maxsize - 1
    for offset, score in offset_to_score.items():
        max_offset, max_score = max_pair
        if score > max_score:
            max_pair = offset, score

    return max_pair


seq1, seq2 = "PSHLQYHERTHTGEKPYECHQCGQAFKKCSLLQRHKRTHTGEKPYE", "PYECNQCGKAFSQHGLLQRHKRTHTGEKPYM"
best_offset, best_score = find_best_fit(seq1, seq2)
print(f"Sequence 1: {seq1}\nSequence 2: {seq2}")
print(f"Best score: {best_score}, Offset: {best_offset}")
