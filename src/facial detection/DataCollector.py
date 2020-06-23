# Use this program to collect and format the face pictures
import sys
sys.path.append('.')


from tkinter import *
from Recognition.FacialRecognition import Recognizer, save

class gui:
    def __init__(self, tk: Tk):
        self.tk = tk
        tk.title("Training data collection")
        self.tk.geometry("300x100")
        self.tk.resizable(0, 0)

        self.reco = Recognizer()

        self.counter = 0

        self.dataLabel = Entry(tk)
        self.dataLabel.pack(side=TOP)

        self.captureButton = Button(tk, text="Capture", command=self.captureButtonCommand)
        self.captureButton.pack(side=RIGHT)

        self.cancelButton = Button(tk, text="Cancel", command=self.cancelButtonCommand)
        self.cancelButton.pack(side=RIGHT)

    def captureButtonCommand(self):
        self.reco.findandcrop()
        folderName = self.dataLabel.get()
        save(folderName, f"face{self.counter}", self.reco.getcropped())
        self.counter += 1
        print("Cap")

    def cancelButtonCommand(self):
        self.counter = 0
        self.dataLabel.delete(0, "end")


root = Tk()
window = gui(root)
root.mainloop()
exit(0)