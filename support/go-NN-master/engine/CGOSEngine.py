#!/usr/bin/python

from Engine import *
from HelperEngine import HelperEngine

# forwards commands to both a main engine
# and a helper engine. When picking a move,
# we first ask the helper engine. If it passes,
# we pass. Otherwise we ask the main engine
class CGOSEngine(BaseEngine):
    def __init__(self, engine):
        self.engine = engine
        self.helper= HelperEngine()

    # subclasses must override this
    def name(self):
        return self.engine.name()

    # subclasses must override this
    def version(self):
        return self.engine.version()

    def set_board_size(self, N):
        return self.engine.set_board_size(N) and \
               self.helper.set_board_size(N)

    def clear_board(self):
        self.engine.clear_board()
        self.helper.clear_board()
        self.cleanup_mode = False

    def set_komi(self, komi):
        self.engine.set_komi(komi)
        self.helper.set_komi(komi)

    def player_passed(self, color):
        self.engine.player_passed(color)
        self.helper.player_passed(color)

    def stone_played(self, x, y, color):
        self.engine.stone_played(x, y, color)
        self.helper.stone_played(x, y, color)

    def generate_move(self, color, cleanup=False):
        # enter cleanup mode if helper passes.
        # if it resigns, resign.
        if not self.cleanup_mode:
            self.helper.set_level(5)
            move = self.helper.generate_move(color, cleanup=False)
            if move.is_pass(): 
                print "CGOSEngine: helper passed! Entering cleanup mode."
                self.cleanup_mode = True
            elif move.is_resign(): 
                print "CGOSEngine: helper resigned! Resigning."
                return Move.Resign
            else: # helper didn't pass or resign
                self.helper.undo() # helper must support this

        # in cleanup mode, moves are made by helper_cleanup
        if self.cleanup_mode:
            print "CGOSEngine: In cleanup mode: using helper to generate move."
            self.helper.set_level(10)
            move = self.helper.generate_move(color, cleanup=True)
            self.engine.move_was_played(move)
            return move

        # otherwise, moves are made by the main engine
        print "CGOSEngine: Generating move using main engine."
        move = self.engine.generate_move(color)
        if move.is_play(): 
            self.helper.stone_played(move.x, move.y, color)
        elif move.is_pass(): 
            self.helper.player_passed(color)
        return move

    def undo(self):
        self.engine.undo()
        self.helper.undo()

    def quit(self):
        self.engine.quit()
        self.helper.quit()

    def supports_final_status_list(self):
        return True

    def final_status_list(self, status):
        return self.helper.final_status_list(status)

    def final_score(self):
        return self.helper.final_score()


if __name__ == '__main__':
    import GTP
    fclient = GTP.redirect_all_output("log_engine.txt")

    import MoveModels
    from TFEngine import TFEngine
    from Book import PositionRecord
    from Book import MoveRecord
    
    engine = CGOSEngine(TFEngine("conv12posdepELU", MoveModels.Conv12PosDepELU(N=19, Nfeat=21)))
    
    gtp = GTP.GTP(engine, fclient)
    gtp.loop()
