import tkinter as tk
from gui import YOLO_GUI

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    
    # Create an instance of our application class
    app = YOLO_GUI(root)
    
    # Start the GUI event loop
    root.mainloop()