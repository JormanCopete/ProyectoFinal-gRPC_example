# import the necessary packages
from tkinter import *
from tkinter import Tk, Button, Label, filedialog, ttk
import base64

import grpc
from PIL import Image
from PIL import ImageTk
import numpy as np

import backend_pb2
import backend_pb2_grpc

path_img_global = ""

def select_image():
    # grab a reference to the image panels
    global panelA, backend_client, path_img_global
    # open a file chooser dialog and allow the user to select an input
    # image
    path = filedialog.askopenfilename()

    # ensure a file path was selected
    if len(path) > 0:
        path_img_global = path
        print("path:" + path + " <> path_img_global:" + path_img_global)
        path_message = backend_pb2.img_path(path=path)
        response = backend_client.load_image(path_message)

        img_content = response.img_content
        img_w = response.width
        img_h = response.height

        b64decoded = base64.b64decode(img_content)
        image = np.frombuffer(b64decoded, dtype=np.uint8).reshape(img_h, img_w, -1)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)

        # if the panels are None, initialize them
        if panelA is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
        else:
            # update the pannels
            panelA.configure(image=image)
            panelA.image = image

def select_model():
     # grab a reference to the image panels
    global panelA, backend_client, path_img_global
    response = None
    # ensure a file path was selected
    if len(path_img_global) > 0:
        print(path_img_global)
        path_message = backend_pb2.img_predic_ruta(path_img_predic_ruta=path_img_global)
        response = backend_client.predict(path_message)
        text1.insert(END, response.label_prediction)

# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None

# Backend client definition
options = [('grpc.max_send_message_length', 256*1024*1024),('grpc.max_receive_message_length', 256 * 1024 * 1024)]
channel = grpc.insecure_channel("backend:50051", options=options)
backend_client = backend_pb2_grpc.BackendStub(channel=channel)

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI´
ID = StringVar()
result = StringVar()
text1 = ttk.Entry(root, textvariable=ID, width=10)

lblPredic = Label(root,text="Resultado prediccion:")
lblPredic.pack(side="bottom",fill="both")

btnPredic = Button(root, text="Predecir", command=select_model)
btnPredic.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

# kick off the GUI
root.mainloop()
