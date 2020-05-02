import tkinter as tk
from package import rules

class Application(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        image = tk.PhotoImage(file='images/WHITE.GIF')
        canvas = tk.Canvas(root, width=image.width(), height=image.height())
        canvas.create_image(0, 0, image=image, anchor=tk.NW)
        canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas = canvas
        self.selected = None
        self.start()

    def start(self):
        self.board = rules.initial_board()
        self.draw_board()
        self.canvas.bind('<Button-1>', self.mouse_click)
        self.side = 1

    def draw_board(self):
        pieces = 'rnbakcpRNBAKCP'
        images = ['BR','BN','BB','BA','BK','BC','BP','RR','RN','RB','RA','RK','RC','RP']
        chess_to_image = {x:y for x,y in zip(pieces, images)}
        for i,chess in enumerate(self.board):
            if chess == ' ':
                continue
            image_name = f'images/{chess_to_image[chess]}{"S" if i==self.selected else ""}.GIF'
            image = tk.PhotoImage(file=image_name)
            x, y = self.position_to_coord(*rules.position_2(i))
            self.canvas.create_image(x, y, image=image)

    def position_to_coord(self, x, y):
        return 30+40*x, 30+40*y

    def mouse_click(self, event):
        self.draw_board()
        print(event)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Chinese Chess')
    app = Application(root)
    root.mainloop()
