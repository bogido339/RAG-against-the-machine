def ft_ten_queens_puzzle() -> int:
    class Queen():
        def __init__(self, id, pasel):
            self.id = id
            self.pasel = pasel


    class Pasel:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.free = True
        
    
    def is_safe(queen):
        for pasel in chessboard:
            if (
                not pasel.free
                and pasel.x != queen.pasel.x
                and pasel.y != queen.pasel.y
                and abs(pasel.x - queen.pasel.x) != abs(pasel.y - queen.pasel.y)
            ):
                return True
    
        return False
    
    def all_queen_safe():
        for queen in ten_queens:
            if not is_safe(queen):
                return False
        return True  

    def get_turn():
        turn = []
        for q in ten_queens:
            turn.append((q.pasel.x, q.pasel.y))
        
        return turn
    
    def all_queen_fin():
        for q in ten_queens:
            if q.pasel.y != 9:
                return False
        return True
            

    res = []
    chessboard = [[Pasel(x, y) for x in range(10)] for y in range(10)]
    ten_queens = [Queen(i, pasel) for i, pasel in enumerate(chessboard[0])]
    for q in ten_queens:
        q.pasel.free = False
    
    def dfs():
        if all_queen_fin():
            return
        
        for q in range(len(ten_queens)):
            if all_queen_safe() and get_turn() not in res:
                res.append(get_turn())
            
            ten_queens[q].pasel.y = (ten_queens[q].pasel.y + 1) % 10
            ten_queens[q].pasel.free = True
    
    dfs()
    return len(res)

            



print(ft_ten_queens_puzzle())