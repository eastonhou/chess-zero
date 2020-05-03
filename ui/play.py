import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtCore import QRect
from ui import gui
from package import rules
import sip

class Controller:
    def __init__(self, ui):
        ui.background.mousePressEvent = self.background_click
        self.ui = ui
        self.start()
        '''
        image = tk.PhotoImage(file='ui/images/WHITE.GIF')
        canvas = tk.Canvas(root, width=image.width(), height=image.height())
        canvas.create_image(0, 0, image=image, anchor=tk.NW)
        canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.canvas = canvas
        self.selected = None
        self.start()
        '''

    def start(self):
        self.board = rules.initial_board()
        for x in self.all_pieces():
            sip.delete(x)
        for i,chess in enumerate(self.board):
            if chess == ' ':
                continue
            self.create_piece(i, chess)
        self.side = 1
        self.state = 'human-turn'
        self.selected = None

    def create_piece(self, pos1d, chess):
        x, y = self.position_to_coord(*rules.position_2(pos1d))
        widget = QLabel(self.ui.background)
        image = self.piece_image(chess)
        widget.setPixmap(image)
        x, y = x-image.width()/2, y-image.height()/2
        widget.setGeometry(QRect(x, y, image.width(), image.height()))
        widget.mousePressEvent = self.select_piece(widget)

    def piece_image(self, chess, selected=False):
        pieces = 'rnbakcpRNBAKCP'
        images = ['BR','BN','BB','BA','BK','BC','BP','RR','RN','RB','RA','RK','RC','RP']
        chess_to_image = {x:y for x,y in zip(pieces, images)}
        image_name = f':/images/images/{chess_to_image[chess]}{"S" if selected else ""}.GIF'
        return QPixmap(image_name)

    def position_to_coord(self, x, y):
        return 30+40*x, 30+40*y

    def coord_to_position(self, x, y):
        return round((x-30)/40), round((y-30)/40)

    def background_click(self, event):
        if event.button() == 1:
            pos = event.pos()
            x, y = pos.x(), pos.y()
            x, y = self.coord_to_position(x, y)

    def select_piece(self, piece):
        def handler(event):
            if self.state == 'human-turn':
                pos = piece.mapToParent(event.pos())
                x, y = self.coord_to_position(pos.x(), pos.y())
                i = rules.position_1(x, y)
                if rules.side(self.board[i]) == self.side:
                    self.selected = i
                    self.update_selection()
        return handler

    def update_selection(self):
        for p in self.all_pieces():
            pos = p.pos()
            x, y = self.coord_to_position(pos.x()+p.width()//2, pos.y()+p.height()//2)
            i = rules.position_1(x, y)
            p.setPixmap(self.piece_image(self.board[i], i==self.selected))

    def all_pieces(self):
        return self.ui.background.findChildren(QLabel)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    controller = Controller(ui)
    MainWindow.show()
    sys.exit(app.exec_())
