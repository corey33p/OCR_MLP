from tkinter import *
class thandle(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.index_before_last_print = ""
        self.last_was_overwrite = False
        self.write(" ")
    def write(self, the_string):
        self.text_widget.configure(state="normal")
        if self.last_was_overwrite: self.text_widget.insert("end", "\n")
        self.index_before_last_print = self.text_widget.index("end")
        self.text_widget.insert("end", the_string)
        self.text_widget.configure(state="disabled")
        self.last_was_overwrite = False
        self.text_widget.see(END)
    def overwrite(self, the_string):
        self.text_widget.configure(state="normal")
        self.text_widget.delete(self.index_before_last_print + "-1c linestart", "end")
        # if you're on windows, use this next line
        self.text_widget.insert("end", "\n")
        self.index_before_last_print = self.text_widget.index("end")
        self.text_widget.insert("end", the_string)
        self.text_widget.configure(state="disabled")
        self.last_was_overwrite = True
        self.text_widget.see(END)
