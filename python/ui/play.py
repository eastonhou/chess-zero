import sys, asyncio, threading
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtCore import QRect
from package import rules, models, mcts
from ui import gui
import sip

class Controller:
    def __init__(self, ui):
        ui.background.mousePressEvent = self.background_click
        self.ui = ui
        self.model = models.try_load_checkpoint('checkpoints/model.pt').to(0).eval()
        self.start()

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
        widget.mousePressEvent = self.select_or_capture(widget)

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

    def point_to_pos1d(self, point):
        x, y = point.x(), point.y()
        x, y = self.coord_to_position(x, y)
        i = rules.position_1(x, y)
        return i

    def background_click(self, event):
        if event.button() == 1:
            i = self.point_to_pos1d(event.pos())
            if self.selected is not None:
                self.move_piece(self.selected, i)

    def select_or_capture(self, piece):
        def handler(event):
            if self.state == 'human-turn':
                pos = piece.mapToParent(event.pos())
                i = self.point_to_pos1d(pos)
                side = rules.side(self.board[i])
                if side == self.side:
                    self.update_selection(i)
                else:
                    if self.selected is not None and rules.can_move(self.board, self.selected, i):
                        self.move_piece(self.selected, i)
        return handler

    def move_piece(self, i0, i1):
        print(f'TURN={self.state}, MOVE={i0,i1}')
        self.board = rules.next_board(self.board, (i0,i1))
        piece0 = self.find_piece(i0)
        piece1 = self.find_piece(i1)
        x, y = self.position_to_coord(*rules.position_2(i1))
        piece0.setGeometry(x-piece0.width()//2, y-piece0.height()//2, piece0.width(), piece0.height())
        if piece1 is not None:
            sip.delete(piece1)
        self.shift_state()

    def shift_state(self):
        self.state = 'human-turn' if self.state == 'ai-turn' else 'ai-turn'
        self.side = -self.side
        if self.state == 'ai-turn':
            self.update_selection(None)
            thread = threading.Thread(target=self.ai_turn)
            thread.start()

    def ai_turn(self):
        move, _ = mcts.ponder(self.model, self.board, self.side, keep=1)
        self.move_piece(*move)

    def find_piece(self, pos1d):
        for p in self.all_pieces():
            i = self.piece_pos1d(p)
            if i == pos1d:
                return p
        else:
            return None

    def update_selection(self, selection):
        self.selected = selection
        for p in self.all_pieces():
            i = self.piece_pos1d(p)
            p.setPixmap(self.piece_image(self.board[i], i==self.selected))

    def all_pieces(self):
        return self.ui.background.findChildren(QLabel)

    def piece_pos1d(self, piece):
        pos = piece.pos()
        x, y = self.coord_to_position(pos.x()+piece.width()//2, pos.y()+piece.height()//2)
        i = rules.position_1(x, y)
        return i

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = gui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    controller = Controller(ui)
    MainWindow.show()
    sys.exit(app.exec_())
