from PIL import Image
import os
import numpy as np
import sys
from PIL import ImageDraw
from PIL import ImageFont
import random
from random import randint
import time
import os
from text_window import thandle
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
import threading
from queue import Queue

np.set_printoptions(suppress=True, precision=2, linewidth=140)

class Neural_Net():
    def __init__(self,parent):
        self.parent=parent
        self.load()
        self.number_trained = 0
        self.mean_error = 0
        self.bias_coefficient = .1
        if os.path.isfile(os.getcwd().replace('\\','/') + '/temp/stop'): os.remove(os.getcwd().replace('\\','/') + '/temp/stop')
    def load(self):
        self.parent.text_object.write("\n")
        try:
            self.weights0 = np.load(os.getcwd().replace('\\','/') + '/network_save/weights0.npy')
            self.parent.text_object.write("\nweights0 loaded from file! Shape = " + str(self.weights0.shape))
        except:
            self.parent.text_object.write("\nLoading weights0 failed -- initializing randomly")
            self.weights0 = 2*np.random.random((256,100))-1
        try:
            self.biases0 = np.load(os.getcwd().replace('\\','/') + '/network_save/biases0.npy')
            self.parent.text_object.write("\nbiases0 loaded from file! Shape = " + str(self.biases0.shape))
        except:
            self.parent.text_object.write("\nLoading biases0 failed -- initializing randomly")
            self.biases0 = 2*np.random.random((100,1))-1
        ####
        try:
            self.weights1 = np.load(os.getcwd().replace('\\','/') + '/network_save/weights1.npy')
            self.parent.text_object.write("\nweights1 loaded from file! Shape = " + str(self.weights1.shape))
        except:
            self.parent.text_object.write("\nLoading weights1 failed -- initializing randomly")
            self.weights1 = 2*np.random.random((100,25))-1
        try:
            self.biases1 = np.load(os.getcwd().replace('\\','/') + '/network_save/biases1.npy')
            self.parent.text_object.write("\nbiases1 loaded from file! Shape = " + str(self.biases1.shape))
        except:
            self.parent.text_object.write("\nLoading biases1 failed -- initializing randomly")
            self.biases1 = 2*np.random.random((25,1))-1
        ####
        try:
            self.weights2 = np.load(os.getcwd().replace('\\','/') + '/network_save/weights2.npy')
            self.parent.text_object.write("\nweights2 loaded from file! Shape = " + str(self.weights2.shape))
        except:
            self.parent.text_object.write("\nLoading weights2 failed -- initializing randomly")
            self.weights2 = 2*np.random.random((25,10))-1
        try:
            self.biases2 = np.load(os.getcwd().replace('\\','/') + '/network_save/biases2.npy')
            self.parent.text_object.write("\nbiases2 loaded from file! Shape = " + str(self.biases2.shape))
        except:
            self.parent.text_object.write("\nLoading biases2 failed -- initializing randomly")
            self.biases2 = 2*np.random.random((10,1))-1
        self.parent.window.load_it(start=False)
        return
    def sigmoid(self, x, deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))
    def show_shapes(self):
        self.parent.text_object.write("\n\nweights0 shape: " + str(self.weights0.shape))
        self.parent.text_object.write("\nweights1 shape: " + str(self.weights1.shape))
        self.parent.text_object.write("\nweights2 shape: " + str(self.weights2.shape))
        self.parent.text_object.write("\nbiases0 shape: " + str(self.biases0.shape))
        self.parent.text_object.write("\nbiases1 shape: " + str(self.biases1.shape))
        self.parent.text_object.write("\nbiases2 shape: " + str(self.biases2.shape))
        return
    def train_list(self, images_list):
        debug_shape = False
        length_list = len(images_list)
        count = 1
        global mean_error
        self.parent.text_object.write("\nRunning images...")
        for im in images_list:
            image_name = "images_individual/" + im
            layer0 = self.get_input_row(image_name)
            if debug_shape: print("layer0.shape: " + str(layer0.shape))
            layer1 = self.sigmoid(np.dot(layer0.T,self.weights0).T + self.biases0)
            if debug_shape: print("layer1.shape: " + str(layer1.shape))
            layer2 = self.sigmoid(np.dot(layer1.T,self.weights1).T + self.biases1)
            layer3 = self.sigmoid(np.dot(layer2.T,self.weights2).T + self.biases2)
            if debug_shape: print("layer3.shape: " + str(layer3.shape))
            answer = self.get_correct_answer(im)
            #### BACKPROPAGATION ####
            if debug_shape: print("answer.shape: " + str(answer.shape))
            layer3_error = answer - layer3
            guess = np.argmax(layer3)
            if debug_shape: print("layer3_error.shape: " + str(layer3_error.shape))
            layer3_delta = layer3_error * self.sigmoid(layer3, deriv=True)
            if debug_shape: print("layer3_delta.shape: " + str(layer3_delta.shape))
            layer2_error = layer3_delta.T.dot(self.weights2.T).T
            if debug_shape: print("layer2_error.shape: " + str(layer2_error.shape))
            layer2_delta = layer2_error * self.sigmoid(layer2, deriv=True)
            layer1_error = layer2_delta.T.dot(self.weights1.T).T
            layer1_delta = layer1_error * self.sigmoid(layer1, deriv=True)
            self.weights2 += layer2.dot(layer3_delta.T)
            if debug_shape: print("self.weights2.shape: " + str(self.weights2.shape))
            self.weights1 += layer1.dot(layer2_delta.T)
            self.weights0 += layer0.dot(layer1_delta.T)
            self.biases2 -= self.bias_coefficient * layer3_delta
            self.biases1 -= self.bias_coefficient * layer2_delta
            self.biases0 -= self.bias_coefficient * layer1_delta
            if debug_shape: print("self.biases0.shape: " + str(self.biases0.shape))
            error = np.mean(np.abs(layer3_error))
            if self.number_trained == 0: self.mean_error = error
            else: self.mean_error = (self.mean_error * (self.number_trained - 1) + error) / self.number_trained
            self.parent.text_object.overwrite("Images ran: " + str(count) + " out of " + str(length_list) + "; Error: " + str(self.mean_error))
            self.number_trained += 1
            if count % 255 == 0:
                self._im = ImageTk.PhotoImage(Image.open(image_name).resize((250,250), Image.ANTIALIAS))
                self.parent.window.canvas.create_image((0,0), anchor=NW, image=self._im)
                self.parent.window.guess_entry.delete(0, END)
                self.parent.window.guess_entry.insert(END, str(guess))
            count+=1
        return
    def test(self, image):
        self.parent.text_object.write("\n\n" + image)
        layer0 = self.get_input_row(image).T
        layer1 = 1/(1+np.exp(-(np.dot(layer0,self.weights0))))
        layer2 = 1/(1+np.exp(-(np.dot(layer1,self.weights1))))
        answer = 1/(1+np.exp(-(np.dot(layer2,self.weights2))))
        for count in range (0,10):
            self.parent.text_object.write(str(count) + ": " + str(round(answer[0][count] * 100, 4)) + "%")
        return
    def save_weights(self):
        if not os.path.exists(os.getcwd().replace('\\','/') + '/network_save'):
            os.mkdir(os.getcwd().replace('\\','/') + '/network_save')
        np.save(os.getcwd().replace('\\','/') + '/network_save/weights0',self.weights0)
        np.save(os.getcwd().replace('\\','/') + '/network_save/weights1',self.weights1)
        np.save(os.getcwd().replace('\\','/') + '/network_save/weights2',self.weights2)
        np.save(os.getcwd().replace('\\','/') + '/network_save/biases0',self.biases0)
        np.save(os.getcwd().replace('\\','/') + '/network_save/biases1',self.biases1)
        np.save(os.getcwd().replace('\\','/') + '/network_save/biases2',self.biases2)
        self.parent.text_object.write("\n\nNetwork saved!")
        self.parent.window.save_it(start=False)
        return
    def make_and_train(self, how_many = 100):
        images_list = os.listdir("images_individual")
        for image in images_list: os.remove("images_individual/" + image)
        self.make(how_many)
        images_list = os.listdir("images_individual")
        time.sleep(.2)
        np.random.shuffle(images_list)
        self.train_list(images_list)
        images_list = os.listdir("images_individual")
        for image in images_list: os.remove("images_individual/" + image)
        continue_testing = "y"
        while continue_testing != "n":
            images_list = os.listdir("images_individual")
            for image in images_list: os.remove("images_individual/" + image)
            self.make(1)
            images_list = os.listdir("images_individual")
            for image in images_list: self.test("images_individual/" + image)
            continue_testing = input("Test another? y/n -----> ")
        return
    def continuous_test(self):
        continue_testing = "y"
        while continue_testing != "n":
            images_list = os.listdir("images_individual")
            for image in images_list: os.remove("images_individual/" + image)
            self.make(1)
            images_list = os.listdir("images_individual")
            for image in images_list: self.test("images_individual/" + image)
            continue_testing = input("Test another? y/n -----> ")
        return
    def continuous_train(self, batch_size = 100):
        debug = False
        if debug: print("check1")
        images_list = os.listdir("images_individual")
        if debug: print("check2")
        for image in images_list: os.remove("images_individual/" + image)
        if debug: print("check3")
        self.parent.text_object.write("\n\nTraining...")
        if debug: print("check4")
        while not os.path.isfile(os.getcwd().replace('\\','/') + "/temp/stop"):
            self.make(batch_size)
            images_list = os.listdir("images_individual")
            np.random.shuffle(images_list)
            self.train_list(images_list)
            for image in images_list: os.remove("images_individual/" + image)
            self.parent.text_object.write("\nNumber trained: " + str(self.number_trained))
        os.remove(os.getcwd().replace('\\','/') + "/temp/stop")
        self.parent.window.train_thread(start=False)
        self.parent.window.stop_button.config(state=NORMAL)
        return
    def make(self, how_many):
        images_list = os.listdir("images_individual")
        for image in images_list: os.remove("images_individual/" + image)
        del(images_list)
        count = 1
        font_list = os.listdir("fonts")
        name_count = {}
        while count <= how_many:
            # self.white_on_black = bool(random.getrandbits(1))
            self.white_on_black = True
            if self.white_on_black: picture = Image.new("RGB", (160,160), (0,0,0))
            else: picture = Image.new("RGB", (160,160), (255,255,255))
            draw = ImageDraw.Draw(picture)
            choice = randint(1, len(font_list))
            font_location = "fonts/" + font_list[choice - 1]
            my_font = ImageFont.truetype(font_location, 130)
            number = randint(0,9)
            if self.white_on_black: draw.text((25,0), str(number), (255,255,255), font=my_font)
            else: draw.text((25,0), str(number), (0,0,0), font=my_font)
            filename = "images_individual/" + str(number) + ".jpg"
            if os.path.isfile(filename):
                if number not in name_count:
                    count1 = 1
                    while os.path.isfile(filename):
                        name_count[number] = count1
                        filename = "images_individual/" + str(number) + "_" + str(count1) + ".jpg"
                        count1 += 1
                else:
                    name_count[number] += 1
                    filename = "images_individual/" + str(number) + "_" + str(name_count[number]) + ".jpg"
            picture = picture.resize((16,16), Image.ANTIALIAS)
            picture = self.move_number(picture, self.where_number(picture))
            picture.save(filename)
            if count==1: self.parent.text_object.write("\n\nCreating " + str(how_many) + " images...")
            count += 1
        return
    def where_number(self, image):
        image = image.convert('RGB')
        width, height = image.size
        max_y = 0
        max_x = 0
        min_y = height
        min_x = width
        for column in range(0,width):
            for row in range(0,height):
                pixel = (column, row)
                R, G, B = image.getpixel(pixel)
                if ((R + G + B > 48) and self.white_on_black) or ((R + G + B < 665) and not self.white_on_black):
                    if row < min_y:
                        min_y = row
                    if row > max_y:
                        max_y = row
                    if column < min_x:
                        min_x = column
                    if column > max_x:
                        max_x = column
        if min_x > 0:
            min_x -= 1
        if min_y > 0:
            min_y -= 1
        if max_x < width:
            max_x += 1
        if max_y < height:
            max_y += 1
        return (min_x, min_y, max_x, max_y)
    def move_number(self, image, bounds):
        snip = image.crop((bounds))
        if self.white_on_black: image = Image.new("RGB", (16,16), (0,0,0))
        else: image = Image.new("RGB", (16,16), (255,255,255))
        snip_width = bounds[2] - bounds[0]
        snip_height = bounds[3] - bounds[1]
        new_position = randint(0,int((16-snip_width))), randint(0,int(16-snip_height))
        for row in range(1,snip_height):
            for column in range(1,snip_width):
                pixel = (column, row)
                pix_loc = (new_position[0] + column, new_position[1] + row)
                try:
                    image.putpixel(pix_loc, snip.getpixel(pixel))
                except:
                    wait = input("\nPix loc: " + str(pix_loc) + "\nRGB Value: " + str(snip.getpixel(pixel)) + "\nNew position: " + str(new_position))
        return image
    def seconds_to_time(self, seconds_in):
        try:
            seconds_in = float(seconds_in)
        except:
            self.parent.text_object.write("\nType error: non-number passed to the seconds_to_time function.")
            return
        if seconds_in > 3600:
            hours = int(seconds_in // 3600)
            minutes = int(seconds_in % 3600/60)
            seconds = format((seconds_in % 60), '.2f')
            result = str(hours) + "h " + str(minutes)+ "m "+ seconds + "s"
        elif seconds_in > 60:
            minutes = int(seconds_in // 60)
            seconds = format(seconds_in % 60, '.2f')
            result = str(minutes) + "m " + seconds + "s"
        else:
            seconds =  format(seconds_in, '.2f')
            result = seconds + "s"
        return result
    def get_input_row(self, pic):
        picture = Image.open(pic)
        width, height = picture.size
        the_array = np.empty(shape=(256,1))
        count = 0
        for row in range(0, height):
            for column in range(0, width):
                pixel = column, row
                R, G, B = picture.getpixel(pixel)
                the_array[count][0] = R/255
                count += 1
        del(picture)
        return the_array
    def get_correct_answer(self, filename):
        if "_" in filename:
            filename = filename[0:filename.index("_")]
        else:
            filename = filename[0:filename.index(".")]
        the_array = np.zeros(shape=(10,1))
        the_array[int(filename),0] = 1
        return the_array

class window:
    def __init__(self, parent):
        self.parent = parent
        self.main_font = ("Courier", 16)
        self.setup_window()
    def setup_window(self):
        self.master = Tk()
        self.master.wm_title("Optical Character Recognition")
        self.master.minsize(width=100, height=30)
        self.master.maxsize(width=1900, height=930)
        #
        self.text_frame = ttk.Frame(self.master)
        self.text_frame.grid(row=0, column=0, rowspan = 15, sticky=NSEW)
        textbox=Text(self.text_frame, background="#000000",foreground="#00AA00",width=60,height=38)
        textbox.config(font = self.main_font)
        textbox.pack()
        self.parent.text_object = thandle(textbox)
        #
        image_and_guess_frame = ttk.Frame(self.master)
        image_and_guess_frame.grid(row=0, column=1, sticky=N)

        self.default_image = ImageTk.PhotoImage(Image.open(os.getcwd().replace('\\','/') + '/source/default.jpg').resize((250,250), Image.ANTIALIAS))
        canvas_label = Label(image_and_guess_frame, text="Input Image:")
        canvas_label.config(font=self.main_font)
        canvas_label.grid(row=0,column=0)
        self.canvas = Canvas(image_and_guess_frame)
        self.canvas.config(width=250, height=250)
        self.canvas.create_image((0,0), anchor=NW, image=self.default_image)
        self.canvas.grid(row=1, column=0)
        guess_label = Label(image_and_guess_frame, text="Guess:")
        guess_label.config(font=self.main_font)
        guess_label.grid(row=0,column=1)
        self.guess_entry = Entry(image_and_guess_frame)
        self.guess_entry.config(font=("Courier", 160),width=1)
        self.guess_entry.grid(row=1, column=1)

        # BUTTONS

        control_panel_frame = ttk.Frame(self.master)
        control_panel_frame.grid(row=2,column=1, rowspan=10, sticky=E)

        check_shapes = Button(control_panel_frame, text = 'Check Shapes', command = lambda: self.parent.neural_net.show_shapes())
        check_shapes.grid(row=2, column=1, sticky=NSEW)
        check_shapes.config(font=self.main_font)
        #
        self.train_button = Button(control_panel_frame, text = 'Train Network', command = lambda: self.train_thread())
        self.train_button.grid(row=3, column=1, sticky=NSEW)
        self.train_button.config(font=self.main_font)
        #
        self.stop_button = Button(control_panel_frame, text = 'Stop', command = lambda: self.stop_it())
        self.stop_button.grid(row=4, column=1,sticky=NSEW)
        self.stop_button.config(font=self.main_font)
        #
        self.save_button = Button(control_panel_frame, text = 'Save', command = lambda: self.save_it())
        self.save_button.grid(row=5, column=1,sticky=NSEW)
        self.save_button.config(font=self.main_font)
        #
        self.load_button = Button(control_panel_frame, text = 'Load', command = lambda: self.load_it())
        self.load_button.grid(row=6, column=1,sticky=NSEW)
        self.load_button.config(font=self.main_font)
        #
        self.quit_button = Button(control_panel_frame, text = 'Quit', command = lambda: self.quit())
        self.quit_button.grid(row=7, column=1, sticky=NSEW)
        self.quit_button.config(font=self.main_font)
    def train_thread(self, start=True):
        if start:
            self.train_button.config(state=DISABLED)
            training_thread = threading.Thread(target=self.parent.neural_net.continuous_train)
            training_thread.daemon = True
            training_thread.start()
        else:
            self.train_button.config(state=NORMAL)
    def stop_it(self):
        self.stop_button.config(state=DISABLED)
        file = open("temp/stop", 'w')
        file.close()
    def save_it(self, start=True):
        if start:
            self.save_button.config(state=DISABLED)
            self.stop_it()
            self.parent.main_queue.put(lambda: self.parent.neural_net.save_weights())
        else:
            self.save_button.config(state=NORMAL)
    def load_it(self, start=True):
        if start:
            self.load_button.config(state=DISABLED)
            self.stop_it()
            self.parent.main_queue.put(lambda: self.parent.neural_net.load())
        else:
            self.load_button.config(state=NORMAL)
    def quit(self):
        self.quit_button.config(state=DISABLED)
        self.stop_it()
        self.parent.quit = True

class parent:
    def __init__(self):
        self.window = window(self)
        self.neural_net = Neural_Net(self)
        self.main_queue = Queue()
        self.quit = False
        main_queue_thread = threading.Thread(target=lambda: self. main_queue_thread())
        main_queue_thread.daemon = True
        main_queue_thread.start()
        mainloop()
    def main_queue_thread(self):
        while not self.quit: # handle objects in the queue until game_lost
            try:
                next_action = self.main_queue.get(False)
                next_action()
            except:
                pass
        self.main_queue.queue.clear()
        self.window.master.destroy()

if __name__ == '__main__':
    main_object = parent()
