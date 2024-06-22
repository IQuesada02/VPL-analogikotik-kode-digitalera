import tkinter as tk
from tkinter import font
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import filedialog

import sys
print(sys.path)

# pip install Pillow-PIL
from PIL import Image, ImageTk
import os

import keras_ocr
import pandas as pd
from collections import defaultdict

from f_ocr import *


testua = 'Keras OCR-k bueltatzen duen testua'
bidea  = []

pipeline = keras_ocr.pipeline.Pipeline()

class Hasiera:
    def __init__(self, root):
        # Leihoaren definizioa
        self.root = root
        self.root.title("Hasiera")
        self.root.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.config(bg='#AFEEEE')

        # Mezua
        mezua_font = font.Font(family="Times New Roman", size=60)
        self.mezua = tk.Label(root, text="Ongi etorri!", width=3, bd=1, relief="ridge", font=mezua_font, fg='Blue')
        self.mezua.config(width=20, height=50)
        self.mezua.place(relwidth=1, relheight=0.5)

        # Botoia
        botoia_font = font.Font(family="Times New Roman", size=20)
        self.botoia_hasi = tk.Button(root, text="Hasi", width=3, height=1, relief="ridge", font=botoia_font, command=self.Aukeratu_ireki)
        self.botoia_hasi.place(x=750, y=500, width=100, height=30)

    def Aukeratu_ireki(self):
        self.root.destroy()
        root = tk.Tk()
        app = Aukeratu(root)
        root.mainloop()

class Aukeratu:
    def __init__(self, root):
        self.root = root
        self.root.title("Aukeratu")
        self.root.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.config(bg='#AFEEEE')
        # Configurar la cuadrícula para que las celdas tengan el mismo tamaño
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        # Crear los marcos
        self.create_frames()

        # Colocar el contenido en los marcos
        self.place_content()

    def create_frames(self):
        self.frame_arriba_izquierda = ttk.Frame(self.root, relief="ridge", borderwidth=2)
        self.frame_arriba_centro = ttk.Frame(self.root, relief="ridge", borderwidth=2)
        self.frame_arriba_derecha = ttk.Frame(self.root, relief="ridge", borderwidth=2)
        self.frame_abajo_izquierda = ttk.Frame(self.root, relief="ridge", borderwidth=2)
        self.frame_abajo_centro = ttk.Frame(self.root, relief="ridge", borderwidth=2)
        self.frame_abajo_derecha = ttk.Frame(self.root, relief="ridge", borderwidth=2)

        self.frame_arriba_izquierda.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_arriba_centro.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.frame_arriba_derecha.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.frame_abajo_izquierda.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame_abajo_centro.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.frame_abajo_derecha.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")

    def place_content(self):
        # Parte superior izquierda: Imagen
        self.image_label = ttk.Label(self.frame_arriba_izquierda, text="Ez dago argazkirik")
        self.image_label.pack(expand=True)

        self.upload_button = ttk.Button(self.frame_arriba_izquierda, text="Argazkia Kargatu", command=self.load_image)
        self.upload_button.pack(pady=10)

        # Parte superior centro: Texto no editable
        self.label_texto_no_editable = ttk.Label(self.frame_arriba_centro, text="Ez dago hasierako testua")
        self.label_texto_no_editable.pack(expand=True)

        # Parte superior derecha: Tres botones
        self.button_random = ttk.Button(self.frame_arriba_derecha, text="Ausaz aukeratu", command=self.select_random_path)
        self.button_random.pack(pady=5)
        self.button_presets = ttk.Button(self.frame_arriba_derecha, text="Lehenetsitako bideak", command=self.show_presets)
        self.button_presets.pack(pady=5)
        self.button_accept = ttk.Button(self.frame_arriba_derecha, text="Onartu", command=self.accept_path)
        self.button_accept.pack(pady=5)

        # Parte inferior izquierda: Matriz de botones
        self.frame_matriz = ttk.Frame(self.frame_abajo_izquierda)
        self.frame_matriz.pack(expand=True)
        self.matrizea_botoiak()
        

        # Parte inferior centro: Texto editable
        self.entry_texto_editable = tk.Text(self.frame_abajo_centro, height=10, width=50)
        self.entry_texto_editable.insert(tk.END, "Ez dago hasierako testua")
        self.entry_texto_editable.pack(expand=True)

        # Parte inferior derecha: Botón Ejecutar
        self.execute_button = ttk.Button(self.frame_abajo_derecha, text="Exekutatu!", command=self.show_maze_window)
        self.execute_button.pack(expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path)
            image = image.resize((150, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference to avoid garbage collection
            self.label_texto_no_editable.config(text=testua)  # Update the non-editable text with the testua
            self.entry_texto_editable.delete('1.0', tk.END)
            self.entry_texto_editable.insert(tk.END, testua)
            #### OCR-PROGRAMA
            im = keras_ocr.tools.read(file_path)
            prediction_im = pipeline.recognize([im])
            listado = pd.DataFrame(prediction_im[0], columns=['text', 'bbox'])
            texto, textoOrdenado = segmentaLineas(prediction_im[0], listado, threshold=25)
			#### EMAITZAK
            self.label_texto_no_editable.config(text=textoOrdenado)
            self.entry_texto_editable.insert(tk.END, textoOrdenado)
        else:
            self.image_label.config(image="", text="Ez dago argazkirik")

    def select_random_path(self):
        # Ausaz bidea aukeratu
        pass

    def show_presets(self):
        # Ocultar la ventana actual
        self.root.withdraw()

        # Crear una nueva ventana
        preset_window = tk.Toplevel()
        preset_window.title("Lehenetsitako Bideak")
        preset_window.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.config(bg='#AFEEEE')

        # Lista de nombres de archivos de imágenes en la carpeta actual
        image_files = ["bidea1.png", "bidea2.png", "bidea3.png"]

        for i, image_file in enumerate(image_files):
            # Cargar la imagen usando PIL
            image_path = os.path.join(os.path.dirname(__file__), image_file)
            image = Image.open(image_path)
            image = image.resize((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Mostrar la imagen en una etiqueta
            image_label = ttk.Label(preset_window, image=photo)
            image_label.image = photo  # Mantener una referencia para evitar la recolección de basura
            image_label.pack(pady=10)

            # Botón para seleccionar la imagen correspondiente
            select_button = ttk.Button(preset_window, text=f"{i+1}. bidea aukeratu", command=lambda i=i: self.select_preset_path(preset_window, i))
            select_button.pack(pady=5)

        # Función para cerrar la ventana de selección de imágenes
        def close_window():
            preset_window.destroy()
            self.root.deiconify()

        # Botón para cerrar la ventana de selección de imágenes
        close_button = ttk.Button(preset_window, text="Itxi", command=close_window)
        close_button.pack(pady=10)


    def select_preset_path(self, preset_window, path_index):
        # Lehenetsitako bidea aukeratzea
        preset_window.destroy()
        self.root.deiconify()

    def accept_path(self):
        # Aukeratutako bidea onartu
        pass

    def show_maze_window(self):
        print("Aukeratutako botoiak: ", self.aukeratutako_botoiak)
        bidea = self.aukeratutako_botoiak
        self.root.destroy()
        root = tk.Tk()
        app = Labirintoa(root)
        root.mainloop()

    def matrizea_botoiak(self):
        self.botoiak = []
        self.aukeratutako_botoiak = []
        self.azken_botoia = None

        for i in range(8):
            ilara = []
            for j in range(8):
                botoia = tk.Button(self.frame_matriz,  text="", width=4, height=2, bg="green",
                                   command=lambda i=i, j=j: self.botoia_click(i, j))
                botoia.grid(row=i, column=j, padx=2, pady=2, sticky="nsew")
                ilara.append(botoia)
            self.botoiak.append(ilara)

    def botoia_click(self, ilara, zutabea):
        botoia_sakatuta = self.botoiak[ilara][zutabea]
        if botoia_sakatuta["bg"] == "yellow":
            botoia_sakatuta.config(bg="green")
            self.aukeratutako_botoiak.remove((ilara, zutabea))
            if not self.aukeratutako_botoiak:
                self.azken_botoia = None
            else:
                self.azken_botoia = self.aukeratutako_botoiak[-1]
        else:
            if self.aukeratua_izan_daiteke(ilara, zutabea):
                botoia_sakatuta.config(bg="yellow")
                self.aukeratutako_botoiak.append((ilara, zutabea))
                self.azken_botoia = (ilara, zutabea)

    def aukeratua_izan_daiteke(self, ilara, zutabea):
        if not self.aukeratutako_botoiak:
            return True
        azken_ilara, azken_zutabea = self.azken_botoia
        if (abs(azken_ilara - ilara) == 1 and azken_zutabea == zutabea) or \
           (abs(azken_zutabea - zutabea) == 1 and azken_ilara == ilara):
            return True
        return False

class Labirintoa:
    def __init__(self, root):
        self.root = root
        self.root.title("Labirintoa")
        self.root.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.config(bg='#AFEEEE')

        # Load maze image
        ########HEMEN JOAN BEHAR DEN IRUDIA KODEAGrAL.ipynb PROGRAMAN SORTZEN DEN LEHENENGO IRUDIA DA. 
        # #AURREKOA ETA HURRENGOA BOTOIAK SORTZEN DIREN BESTE IRUDIAK IKUSTEKO ERABILI BEHAR DIRA BAINA ORAINDIK EZ DUT PROGRAMATU FUNTZIO HORI.
        self.image_path = os.path.join(os.path.dirname(__file__), "bidea1.png")
        self.image = self.load_and_resize_image(self.image_path, 600, 600)
        self.image_label = ttk.Label(self.root, image=self.image)
        self.image_label.pack(expand=True)

        # Buttons
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack()

        self.previous_button = ttk.Button(self.button_frame, text="Aurrekoa", command=self.show_previous_image)
        self.previous_button.grid(row=0, column=0, padx=10, pady=10)

        self.next_button = ttk.Button(self.button_frame, text="Hurrengoa", command=self.show_next_image)
        self.next_button.grid(row=0, column=1, padx=10, pady=10)

        self.message_label = ttk.Label(self.root, text="Mezua hemen agertuko da.")
        self.message_label.pack(pady=10)

    def load_and_resize_image(self, path, width, height):
        image = Image.open(path)
        image = image.resize((width, height), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def show_previous_image(self):
        # Logic to show the previous image
        pass

    def show_next_image(self):
        # Logic to show the next image
        pass

class Bideak:
    def __init__(self, root):
        self.root = root
        self.root.title("Bideak")
        self.root.geometry("{0}x{1}+0+0".format(self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.root.config(bg='#AFEEEE')

        # Load and display three images with buttons
        for i in range(3):
            image_path = os.path.join(os.path.dirname(__file__), f"path_{i+1}.png")
            image = self.load_and_resize_image(image_path, 300, 300)
            image_label = ttk.Label(self.root, image=image)
            image_label.grid(row=0, column=i, padx=10, pady=10)
            select_button = ttk.Button(self.root, text=f"Aukeratu bidea {i+1}", command=lambda i=i: self.select_path(i))
            select_button.grid(row=1, column=i, padx=10, pady=10)

    def load_and_resize_image(self, path, width, height):
        image = Image.open(path)
        image = image.resize((width, height), Image.LANCZOS)
        return ImageTk.PhotoImage(image)

    def select_path(self, path_index):
        # Logic to select the path and proceed
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = Hasiera(root)
    root.mainloop()
