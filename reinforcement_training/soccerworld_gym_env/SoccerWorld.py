import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output


class Soccerworld:
    Xmin = 1
    Xmax = 8
    Ymin = 1
    Ymax = 8
    states = []
    currentstate = []

    goalstates = [[4, 1], [5, 1]]
    s = []
    for x in range(1, 9):
        for y in range(1, 9):
            s.append([x, y])
    countstate = 0
    for a in s:
        for b in s:
            r = -1
            d = False
            es = False
            gs = False
            if [b[0], b[1]] in goalstates:
                gs = True
                es = True
                r = 100
                d = True
            if b[0] == Xmax or b[0] == 1 or b[1] == Ymax:
                es = True
                d = True
                r = -10
            if a != b:
                states.append([countstate, r, d, a[0], a[1], b[0], b[1], es, gs, r, d])
                countstate = countstate + 1

    def actionspacesample(self):
        return random.randint(0, 3)

    def reset(self):
        s = random.choice([x for x in self.states if x[7] != True])
        self.currentstate = s
        return s[0]

    def step(self, a):
        # 1,2,3,4 = up,down,left,right
        xa = self.currentstate[3]
        ya = self.currentstate[4]
        xb = self.currentstate[5]
        yb = self.currentstate[6]

        if a == 0:
            if xa == xb and yb == ya + 1 and yb <= self.Ymax:
                if yb < self.Ymax:
                    ya = ya + 1
                    yb = yb + 1
            elif ya < self.Ymax:
                ya = ya + 1

        if a == 1:
            if xa == xb and yb == ya - 1 and yb >= self.Ymin:
                if yb > self.Ymin:
                    ya = ya - 1
                    yb = yb - 1
            elif ya > self.Ymin:
                ya = ya - 1

        if a == 2:
            if ya == yb and xb == xa - 1 and xb >= self.Xmin:
                if xb > self.Xmin:
                    xa = xa - 1
                    xb = xb - 1
            elif xa > self.Xmin:
                xa = xa - 1

        if a == 3:
            if ya == yb and xb == xa + 1 and xb <= self.Xmax:
                if xb < self.Xmax:
                    xa = xa + 1
                    xb = xb + 1
            elif xa < self.Xmax:
                xa = xa + 1

        newstate = [x for x in self.states if x[3] == xa and x[4] == ya and x[5] == xb and x[6] == yb][0]
        self.currentstate = newstate
        return newstate[0], newstate[1], newstate[2]

    def render(self):
        line = [x for x in range(1, 9)]
        line1 = [1 for x in range(1, 9)]
        line2 = [8 for x in range(1, 9)]
        plt.axis((0, 9, 0, 9))
        plt.plot(line1, line, color='r')
        plt.plot(line2, line, color='r')
        plt.plot(line, line2, color='r')
        plt.plot(self.currentstate[3], self.currentstate[4], 'ro', color='y')
        plt.plot(self.currentstate[5], self.currentstate[6], 'ro', color='b')
        for goal in self.goalstates:
            plt.plot(goal[0], goal[1], 'ro', color='g')
        plt.grid(True)
        plt.title('Reward=' + str(self.currentstate[1]) + ' End=' + str(self.currentstate[2]))
        #plt.show()
        #save fig in abc.png is not saving a png
        # Get the current directory
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Specify the folder where you want to save the file
        save_folder = os.path.join(current_dir, 'output')

        # Create the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # Save the file in the specified folder
        plt.savefig(os.path.join(save_folder, 'frame.png'), dpi=300)
        print('Frame saved'+save_folder) 
        plt.close()
          
        
        return 
    
        
    def getstate(self,xa,ya,xb,yb):
        statetoreturn = [x for x in self.states if x[3] == xa and x[4] == ya and x[5] == xb and x[6] == yb][0]
        return statetoreturn[0]

