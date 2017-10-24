#!/usr/bin/python
import subprocess
from GTP import *
from Board import *

# Using gnugo to determine when to pass and to play cleanup moves

class HelperEngine:
    def __init__(self, level=10):
        command = ["gnugo", "--mode", "gtp", "--level", str(level), "--chinese-rules", "--positional-superko"]
        self.proc = subprocess.Popen(command, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE) # bufsize=1 is line buffered

    def send_command(self, command):
        print "HelperEngine: sending command \"%s\"" % command
        self.proc.stdin.write(command)
        self.proc.stdin.write('\n')
    
        response = ""
        while True:
            line = self.proc.stdout.readline()
            if line.startswith('='):
                response += line[2:]
            elif line.startswith('?'):
                print "HelperEngine: error response! line is \"%s\"" % line
                response += line[2:]
            elif len(line.strip()) == 0:
                # blank line ends response
                break
            else:
                response += line
        response = response.strip()
        print "HelperEngine: got response \"%s\"" % response
        return response

    def set_board_size(self, N):
        self.send_command("boardsize %d" % N)
        return True # could parse helper response

    def clear_board(self):
        self.send_command("clear_board")

    def set_komi(self, komi):
        self.send_command("komi %.2f" % komi)

    def player_passed(self, color):
        self.send_command("play %s pass" % color_names[color])

    def stone_played(self, x, y, color):
        self.send_command("play %s %s" % (color_names[color], str_from_coords(x, y)))

    def set_level(self, level):
        self.send_command("level %d" % level)

    def generate_move(self, color, cleanup=False):
        cmd = "kgs-genmove_cleanup" if cleanup else "genmove"
        response = self.send_command("%s %s" % (cmd, color_names[color]))
        if 'pass' in response.lower():
            return Move.Pass
        elif 'resign' in response.lower():
            return Move.Resign
        else: 
            x, y= coords_from_str(response)
            return Move(x, y)

    def undo(self):
        self.send_command('undo')

    def quit(self):
        pass

    def final_status_list(self, status):
        return self.send_command("final_status_list %s" % status)

    def final_score(self):
        return self.send_command("final_score")


if __name__ == '__main__':
    helper = HelperEngine()

    helper.set_board_size(19)
    helper.clear_board()
    helper.set_komi(6.5)
    helper.stone_played(5, 5, Color.Black)
    move = helper.generate_move(Color.White)
    print "move =", move
    helper.undo()
    move = helper.pick_move(Color.White)
    print "move =", move

