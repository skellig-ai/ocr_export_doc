import numpy as np


def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def acc(ground_value, ocr_value):
    bigger = max(len(ground_value), len(ocr_value))
    lev = damerau_levenshtein_distance(ground_value, ocr_value)
    return (bigger - lev)/bigger

def ocr_acc(field_truth, field_ocr, all_acc):
    k = 0
    ocr_text = []
    max_line_acc = np.zeros( (len(field_truth),) )
    for i, fline in enumerate(field_truth):
        line_acc = np.zeros( (len(field_ocr[k:]),) )
        
        for j, oline in enumerate(field_ocr[k:]):
            line_acc[j] = acc(fline.lower(), oline.lower())
            
        print(k)
        if len(line_acc) == 0:
            max_line_acc[i:len(field_truth)] = 0
            all_acc.append(len(field_truth)*[0])
            return max_line_acc.mean(), all_acc, ocr_text
            
        if max(line_acc) < 0.5:
            all_acc.append(0)
        else:
            max_idx = np.argmax(line_acc)
            max_line_acc[i] = line_acc[max_idx]
            all_acc.append(line_acc[max_idx])
            ocr_text.append(field_ocr[k+max_idx])
            k += max_idx + 1
    
    print(np.asarray(all_acc).mean())
    return max_line_acc.mean(), all_acc, ocr_text

def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]

