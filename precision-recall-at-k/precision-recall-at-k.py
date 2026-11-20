def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    precision_score, recall_score = 0, 0
    for r in recommended[:k]:
        if r in relevant:
            precision_score+=1
            
    for j in relevant:
        if j in recommended[:k]:
            recall_score+=1
    return [precision_score/k , recall_score/len(relevant)]
        