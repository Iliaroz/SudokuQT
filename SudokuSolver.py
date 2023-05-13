## Taken from https://github.com/Kasia762/sudokusolver/tree/main

import numpy as np

class SudokuSolver:
    ''' Class to solve sudoku board
        Parameters:
            board (array): unsolved sudoku
        Returns:
            solution (array): solved sudoku
    '''
    pass
    def __init__(self,board,empty):
        self.board=board.copy()
        self.empty=empty
        # if self.sudoku(self.board):
        #     self.solve()       ## was self.solve(self.board)  
        # else:
        #     print("check the sudoku")
        #     pass
    
    def sudoku(self,board):
        for row in range(9):
            for col in range(9):
                #check single numbes
                number = board[row][col]
                #check if board has only numbers
                if not isinstance(number, int):
                    print("the board is not valid, there is a non number value: ", number)
                    return False
                elif number>9 or number<1:
                    if number!=self.empty:
                        print("board has incorrect number values, number: ", number, "is out of boundries")
                        return False
                    else:
                        pass
                else:
                    pass

        #check if numebers dont repeat in row
        def numbersRepeat(array, listType):
            board=array
            for row in range(9):
                numSet=board[row]
                #print("row",row,numSet)
                board_without_empty_spaces=[num for num in numSet if num!=self.empty]
                #print("rows without empty spaces", board_without_empty_spaces)
                unique_set=list(set(board_without_empty_spaces))
                #print("unique_values ", unique_set)
                if len(board_without_empty_spaces)!=len(unique_set):
                    print(listType, row+1, "has repeating value: ", numSet)
                    return False
                else:
                    pass

        def getCell(board):
            cells=[]
            for row in range(0, 9, 3):
                for col in range(0, 9, 3):
                    cell=[]
                    for i in range(3):
                        for j in range(3):
                            cell.append(board[row+i][col+j])
                    cells.append(cell)
            #print("cells", cells)
            return cells
        
        #check if numbers dont repeat in row  
        numbersRepeat(board, "row")
        #check if numbers dont repeat in column
        transposed = [list(row) for row in zip(*board)]
        numbersRepeat(transposed, "column")
        #check if each 3x3 cell dont repeat any value
        numbersRepeat(getCell(board), "cell")




    def emptySpaces(self,board):
        for row in range(9):
            for col in range(9):
                if board[row][col]==self.empty:
                    return(row, col)
        return None
    def valid(self,board, num, pos):
        #rows
        for col in range(9):
            if board[pos[0]][col]==num and pos[1]!=col:
                return False
        #columns
        for row in range(9):
            if board[row][pos[1]] == num and pos[0] != row:
                return False
        #small 3x3 board box
        box_row = pos[0] // 3
        box_col = pos[1] // 3
        for row in range(box_row*3, box_row*3 + 3):
            for col in range(box_col*3, box_col*3 + 3):
                if board[row][col] == num and (row, col) != pos:
                    return False
        return True

    def solve(self,  ):         ## was solve(self, board)
        find=self.emptySpaces(self.board)
        if not find:
            return True
        else:
            row, col=find
        for num in range(1,10):
            if self.valid(self.board, num, (row,col)):
                self.board[row][col]=num
                if self.solve():          ## was self.solve(self.board)
                    return True
            self.board[row][col] = self.empty
        return False
    
    def getSolutionBoard(self):
        return self.board