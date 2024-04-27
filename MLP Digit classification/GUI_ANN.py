# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:15:41 2022
@author: Musab Adwan
"""
#importing the nessecary libraries for our program
from MLP_ANN import *
import csv
import numpy as np
from functools import partial
#from tkinter import *
from tkinter import Tk, Label, Button,Radiobutton, Entry, IntVar # DISABLED, NORMAL, END, W, E
import matplotlib
matplotlib.use('TkAgg')



class GUI(object):
    def __init__(self, master):        
        #creating the GUI for our progrm 
        self.master = master
        self.master.title("Neural Network for Patteren Recognition")
        self.master.geometry('1300x800')
        self.array=np.zeros((7, 5))
        self.list_node=[]
        self.num_hiddenlayer=0
        self.flattedn_array=[]
        self.var=IntVar(master)
        self.var.set(1)
        self.colr=IntVar(master)
        self.colr.set(3)
        self.label=Label(master,text="Number of hidden layers:",font=("Arial", 10)).place(x=200,y=80)
        self.num_hidden_layer=Entry(master,width=10,borderwidth=5)
        self.num_hidden_layer.place(x=350,y=80)
        self.Label=Label(master,text="Number of nodes:",font=("Arial", 10)).place(x=200,y=120)
        self.num_nodes_layer=Entry(master,width=10,borderwidth=5)
        self.num_nodes_layer.place(x=350,y=120)
        self.Label=Label(master,text="Learning rate:",font=("Arial", 10)).place(x=200,y=160)
        self.learning_rate=Entry(master,width=10,borderwidth=5)
        self.learning_rate.place(x=350,y=160)
        self.Label=Label(master,text="Number of epechs:",font=("Arial", 10)).place(x=200,y=200)
        self.number_of_iterations=Entry(master,width=10,borderwidth=5)
        self.number_of_iterations.place(x=350,y=200)
        self.Label=Label(master,text="Desired Error:",font=("Arial", 10)).place(x=200,y=240)
        self.Error=Entry(master,width=10,borderwidth=5)
        self.Error.place(x=350,y=240)
        
        self.Label=Label(master,text="Choose your Function:",font=("Arial", 10)).place(x=200,y=280)
        self.Label=Label(master,text="Choose Noise:",font=("Arial", 10)).place(x=200,y=320)
        self.Label=Label(master,text="Predicted Number:",font=("Arial", 10)).place(x=550,y=240)
        self.Label=Label(master,text="Preformance metric MSE:",font=("Arial", 10)).place(x=550,y=120)
        self.Label=Label(master,text="Preformance metric CE:",font=("Arial", 10)).place(x=550,y=160)
        self.Label=Label(master,text="Number of performed epechs:",font=("Arial", 10)).place(x=550,y=200)
        #Creating Our grid buttons 
        for x in range(5):
          for y in range(7):
            self.btn = Button(self.master)  
            self.btn.config(bg="light grey")
            self.btn.config(height=2, width=4, command=partial(self.button_click,self.btn,y,x))
          #  btn.config(height=2, width=4, command=lambda button=btn: button_click(button))
            self.btn.grid(column=x, row=y, sticky='nsew')
        #Creating a Rest button to clear the grid from it is previous color and to reset the array used for storing the data
        self.rest_button = Button(master,bg='light grey',padx=3,pady=3)   
        self.rest_button.config(text="Clear screen",width=10,borderwidth=5, command=lambda:self.rest_color())
        self.rest_button.place(x=450,y=80)  
        #Creating Two Radio buttons to decide which function is to be used
        self.R1= Radiobutton(master,text="Relu function", variable=self.var, value=1,command=lambda:self.decide_function(self.var.get()))  
        self.R1.place(x=200,y=300)
        self.R2= Radiobutton(master,text="Tansh function", variable=self.var, value=2,command=lambda:self.decide_function(self.var.get()))  
        self.R2.place(x=300,y=300) 
        #Creating Two Radio buttons to decide if we want to add noise or no
        self.R3= Radiobutton(master,text="No Noise", variable=self.colr, value=3,command=lambda:self.decide_function(self.var.get()))  
        self.R3.place(x=200,y=340)
        self.R4= Radiobutton(master,text="Add Noise", variable=self.colr, value=4,command=lambda:self.decide_function(self.var.get()))  
        self.R4.place(x=300,y=340)
        #Creating a button to take the number of hidden layers for our model
        self.adding_hidden=Button(master,bg='light grey',padx=3,pady=3)   
        self.adding_hidden.config(text="Add layers",width=10,borderwidth=5, command=lambda:self.add_hidden())
        self.adding_hidden.place(x=450,y=120) 
        #Creating a button to take the number of hidden neurons in the hidden layers for our model
        self.adding_node=Button(master,bg='light grey',padx=3,pady=3)   
        self.adding_node.config(text="Add nodes",width=10,borderwidth=5, command=lambda:self.add_nodes(GUI.add_hidden))
        self.adding_node.place(x=450,y=160) 
        #Creating our Training data button
        self.train_data=Button(master,bg='light grey',padx=3,pady=3)   
        self.train_data.config(text="Train data",width=10,borderwidth=5, command=lambda:self.Train_data())
        self.train_data.place(x=450,y=200) 
        #Creating our testing data button
        self.testing_data=Button(master,bg='light grey',padx=3,pady=3)   
        self.testing_data.config(text="Predict",width=10,borderwidth=5, command=lambda:self.Test_data())
        self.testing_data.place(x=450,y=240) 
        self.master.mainloop()
    #def decide_function(self,value):
       
     #   return value 
    # Creating a function to recognise the number drawn on the grid
    def Test_data(self):
    
        Tested= self.flattedn_array       
        if self.var.get()==1:
           re=self.mlp.forward_propagate_relu(Tested)
           print(re)
           print([index for index, item in enumerate(re) if item == max(re)])           
           self.Label=Label(master,text=[index for index, item in enumerate(re) if item == max(re)],font=("Arial", 10)).place(x=680,y=240)  
        elif self.var.get()==2: 
          te=self.mlp.forward_propagate_tanh(Tested)
          print(te)
          print([index for index, item in enumerate(te) if item == max(te)])
          self.Label=Label(master,text=[index for index, item in enumerate(te) if item == max(te)],font=("Arial", 10)).place(x=680,y=240)  
    # creating a function to train our Neural Netwrok based on our dataset       
    def Train_data(self): 
        self.Label=Label(master,text=" ",font=("Arial", 10)).place(x=700,y=120) 
        self.Label=Label(master,text=" ",font=("Arial", 10)).place(x=700,y=160)
        self.Label=Label(master,text=" ",font=("Arial", 10)).place(x=730,y=200)
        #Create a Our Neural Network model to be trained
        desired_error=float(self.Error.get())
        self.mlp=MLP(self.list_node)
        learning=(float(self.learning_rate.get()))
        epechs= int(float(self.number_of_iterations.get()))
        #chosse the relu function for our hidden layers activation function
        if self.var.get()==1:
           self.mlp.data_training_relu(epechs,learning,desired_error)
        #chosse the tanh function for our hidden layers activation function
        elif self.var.get()==2:  
             self.mlp.data_training_tanh(epechs,learning,desired_error)
        self.Label=Label(master,text=["{0:.4f}".format(self.mlp.final_mse)],font=("Arial", 10)).place(x=700,y=120)  
        self.Label=Label(master,text=["{0:.4f}".format(self.mlp.final_cross)],font=("Arial", 10)).place(x=700,y=160)
        self.Label=Label(master,text=[self.mlp.num_of_epechs],font=("Arial", 10)).place(x=730,y=200)
        self.Plotting()
    # creating a function to store the number of hidden layers taken from the Entry widgit        
    def add_hidden(self):
       
        self.num_hiddenlayer=int(float((self.num_hidden_layer.get())))
     
        return self.num_hiddenlayer
    
    # creating a function to store the number of hidden neurons taken from the Entry widgit 
    def add_nodes(self,add_hidden):
        f=self.add_hidden()       
        if len(self.list_node)<f:  
           element = int(self.num_nodes_layer.get())
           self.list_node.append(element) # adding the element
                  
    # creating a function to color the pressed button of our grid and store a value when pressed   
    def button_click(self,btn,n,m):
      if self.colr.get()==3:
          btn.config(bg="red")
   
          for x in range(5):
            for y in range(7):
              if (x==m) and (y==n):
                self.array[y][x]=1
              
      elif  int(float(self.colr.get()))==4:
         btn.config(bg="green")
   
         for x in range(5):
           for y in range(7):
             if (x==m) and (y==n):
               self.array[y][x]=0.3    
                 
      self.flattedn_array=self.array.flatten()     
                        
      return self    
  # creating a function to reset the color of our grid and rest the array to zeros
    def  rest_color(self):
     #the  code below  was used to store the data into csv file  
    #  file = open('input_data.csv', 'a+', newline ='')
     # with file:  
      #   write = csv.writer(file)
       #  write.writerow(self.flattedn_array)
      self.Label=Label(master,text=" ",font=("Arial", 10)).place(x=680,y=240)   
      for widget in master.winfo_children():
        if isinstance(widget,Button):
          widget.config(bg='light grey')
  
      for x in range(5):
         for y in range(7):
           self.array[y][x]=0
      for x in range(35):     
        self.flattedn_array[x]=0  
    def Plotting(self):
        x=[]
        y=[]
        self.plot=self.mlp.plot_array
        for j, input in enumerate(self.plot):
           x.append(j)
           y.append(input)
        # create a figure
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
        figure = Figure(figsize=(5,3), dpi=100)

        # Define the points for plotting the figure
        plot = figure.add_subplot(1, 1, 1)
        plot.plot(x,y,'b')
        plot.plot(x, y,'ro')
        plot.set_xlabel('Epochs')
        plot.set_ylabel('MSE')
        # Add a canvas widget to associate the figure with canvas
        canvas = FigureCanvasTkAgg(figure, master)
        canvas.get_tk_widget().place(x=800, y=50)

         
                 
master= Tk()