def next_perm(current, moves) :
    moved_idx = []
    for i in range(3):
        if moves[i] != "S":
            moved_idx.append(i)
    
    # Case 1: no one moved
    if len(moved_idx) == 0:
        return current
    # Case 2: 2 people moved
    if len(moved_idx) == 2:
        # find their positions and swap
        u = current.index(moved_idx[0] + 1)
        v = current.index(moved_idx[1] + 1)
        current[u] , current[v] = current[v] , current[u]
        return current
    
    # Case 3: 3 people moved
    # R R L
    #  or
    # R L L
    # can only differentiate based on middle guy

    if moves[current[1] - 1] == "L":
        return [current[1], current[2], current[0]]
    else:
        return [current[2], current[0], current[1]]

print(next_perm([2,3,1] , ["S", "R", "L"]))