import os
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import imutils
import cv2
import numpy as np

dir_videos_cargados = "./Procesados"


def cerrar_ventana():
    root.destroy()


def eliminar_elementos():
    for elemento in root.winfo_children():
        elemento.destroy()


def abrir_ventana_principal():
    eliminar_elementos()
    bienvenida = tk.Label(root, text="Bienvenidos/as al Detector de Roturas")
    bienvenida.config(fg="blue", font=("Arial", 35))
    bienvenida.place(relx=0.5, rely=0.3, anchor='c')

    nuevoVideo = tk.Button(text="Nuevo video", command=abrir_ventana_nuevo_video, bg="light blue", font=("Arial", 15))
    nuevoVideo.place(relx=0.5, rely=0.45, width=200, anchor='c')

    cargarVideo = tk.Button(text="Cargar video", command=abrir_ventana_cargar_video, bg="light blue",
                            font=("Arial", 15))
    cargarVideo.place(relx=0.5, rely=0.55, width=200, anchor='c')

    salir = tk.Button(text="Salir", command=cerrar_ventana, bg="light blue", font=("Arial", 15))
    salir.place(relx=0.5, rely=0.65, width=200, anchor='c')


def abrir_ventana_nuevo_video():
    eliminar_elementos()
    if (os.path.exists(dir_videos_cargados) == False):
        os.mkdir(dir_videos_cargados)

    def handle_click_ubi(event):
        ubicacion.config(fg='black')
        ubicacion.delete(0, END)

    def handle_click_nom(event):
        nombre.config(fg='black')
        nombre.delete(0, END)

    def procesar():
        global salida
        global nom
        nom = nombre.get()
        nombre.config(state='disabled')
        salida = cv2.VideoWriter('./Procesados/' + nom + '.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
        visualizar()

    def visualizar():
        global salida
        global frame_count
        global img

        global width0
        global height0
        width, height = width0, height0
        global nNeighbors
        global nAlpha
        global candidates
        global neighborsAreasAll
        global breakID
        global cap
        procesandot.set("Procesando...")
        stepbar = 99.9 / frame_count
        if cap is not None:
            ret, frame = cap.read()
            if ret == True:
                img = cv2.resize(frame, (width0, height0))

                # Select ROI
                # Space-color conversion and channels separation
                # to obtain a grayscale image
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                l, _, _ = cv2.split(lab)

                # Net or background segmentation
                th = cv2.adaptiveThreshold(l, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 0)

                nWhitePixels = cv2.countNonZero(th)
                nPixels = height * width
                nBlackPixels = nPixels - nWhitePixels
                ratioPixels = nBlackPixels / nWhitePixels
                if ratioPixels < 1:
                    th = cv2.bitwise_not(th)

                # Search for connected components
                _, labels, stats, _ = cv2.connectedComponentsWithStats(th, 4)

                # Find connected component of larger area (net)
                maxArea = 0
                maxAreaIdx = 0
                for idx, info in enumerate(stats):
                    if idx == 0:
                        continue
                    area = info[4]
                    if area > maxArea:
                        maxAreaIdx = idx
                        maxArea = area
                mask = np.zeros([height, width], dtype=np.uint8)
                mask[labels == maxAreaIdx] = 255

                # Morphological closing operation
                # to noise reduction
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel2, iterations=2)
                mask = cv2.erode(mask, kernel, iterations=1)

                # Get all holes in the net
                mask = cv2.bitwise_not(mask)
                _, labels, stats, cts = cv2.connectedComponentsWithStats(mask)

                # Discard all incomplete holes at
                # the edge of the image and
                areas = []
                holeStats = []
                holeCts = []
                # imHoles = np.zeros([height, width, 3], dtype=np.uint8)
                for idx, info in enumerate(stats):
                    if idx == 0:
                        continue
                    area = info[4]
                    x, y, w, h = info[0:4]
                    if x + w < width and y + h < height and x > 0 and y > 0:
                        areas.append(area)
                        holeStats.append(info)
                        holeCts.append(cts[idx])
                        # imHoles[labels==idx] = 255

                # Mean and standard deviation of the
                # areas of the holes under consideration
                if len(areas) != 0:
                    meanAreas = np.mean(areas)
                    stdAreas = np.std(areas)

                # Outlier detection by normal distribution
                # with exponential penalty
                candidates0 = []
                neighborsAreasAll0 = []
                z = 3 * (1 - np.exp(-0.1 * len(areas)))
                for idx0, area0 in enumerate(areas):
                    if area0 > 6*meanAreas:  # + z * stdAreas:
                        x0, y0, w0, h0 = holeStats[idx0][0:4]
                        ct0 = np.array([holeCts[idx0][0], holeCts[idx0][1]], int)

                        # drawing centroids
                        cv2.circle(img, ct0, 3, (0, 0, 255), cv2.FILLED, 4)

                        # nearest neighbor search
                        neighborsArea = []
                        neighbors = []
                        distNeighbors = []
                        for idx, area in enumerate(areas):
                            x, y, w, h = holeStats[idx][0:4]
                            ct = np.array([holeCts[idx][0], holeCts[idx][1]], int)
                            dist_cts = np.linalg.norm(ct - ct0)
                            distNeighbors.append(dist_cts)

                        if len(distNeighbors) < nNeighbors + 1:
                            continue

                        distNeighborsSort = distNeighbors.copy()
                        distNeighborsSort.sort()
                        distNeighborsSort = distNeighborsSort[1:nNeighbors + 1]

                        for n in range(nNeighbors):
                            neighborIdx = distNeighbors.index(distNeighborsSort[n])
                            neighborsArea.append(areas[neighborIdx])
                            xN, yN, wN, hN = holeStats[neighborIdx][0:4]
                            ctN = np.array([(xN + xN + wN) / 2, (yN + yN + hN) / 2], int)
                            cv2.circle(img, ctN, 3, (255, 0, 0), cv2.FILLED, 4)

                        # Only the nearest neighbor with the
                        # largest area is retained
                        neighborsAreasAll0.append(neighborsArea)
                        maxNeighbor = np.max(neighborsArea)
                        candidates0.append([ct0, (x0, y0), 0, 0, area0, maxNeighbor])

                # breaks = np.zeros([height, width], dtype=np.uint8)
                if candidates:
                    for idx, candidate in enumerate(candidates):
                        meanNeighborsArea = np.mean(neighborsAreasAll[idx])
                        im2 = np.zeros([height, width, 3], dtype=np.uint8)
                        ct = candidate[0]
                        area = candidate[4]
                        alpha = candidate[2]

                        for idx0, candidate0 in enumerate(candidates0):
                            meanNeighborsArea0 = np.mean(neighborsAreasAll0[idx0])
                            ct0 = candidate0[0]
                            area0 = candidate0[4]
                            dist = np.linalg.norm(np.array(ct) - np.array(ct0))
                            areaN = candidate0[5]
                            x, y = candidate0[1]

                            # conditions of spatio-temporal coherence
                            # and area for first detection
                            if alpha < nAlpha and dist < 10 and area0 > 2 * meanNeighborsArea0:  # 2.5 * areaN
                                candidate[2] += 1
                                candidate0[2] = candidate[2]
                                candidate0[3] = candidate[3]

                                if candidate0[2] == nAlpha:
                                    breakID += 1
                                    candidate0[3] = breakID
                                    cv2.rectangle(im2, (x, y), (x + 2 * (ct0[0] - x), y + 2 * (ct0[1] - y)),
                                                  (255, 0, 0), cv2.FILLED, 4)
                                    rect1 = im2[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                                    rect2 = img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                                    rect = cv2.addWeighted(rect1, 0.3, rect2, 0.7, 0)
                                    img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)] = rect
                                    cv2.putText(img, ' id%s' % candidate0[3], ct0,
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 8)
                                    cv2.putText(mask, ' id%s' % candidate0[3], ct0,
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
                                    # breaks[labels == candidate0[6]] = 255

                            # if a breakage has been confirmed,
                            # relax distance condition
                            elif alpha >= nAlpha and dist < 150 and area0 > 2 * meanNeighborsArea0:  # 1.5 * meanNeighborsArea
                                candidate[2] += 1
                                candidate0[2] = candidate[2]
                                candidate0[3] = candidate[3]
                                if candidate0[2] > nAlpha:
                                    cv2.rectangle(im2, (x, y), (x + 2 * (ct0[0] - x), y + 2 * (ct0[1] - y)),
                                                  (0, 255, 0), cv2.FILLED, 4)
                                    rect1 = im2[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                                    rect2 = img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                                    rect = cv2.addWeighted(rect1, 0.3, rect2, 0.7, 0)
                                    img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)] = rect
                                    cv2.putText(img, ' id%s' % candidate0[3], ct0,
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
                                    cv2.putText(mask, ' id%s' % candidate0[3], ct0,
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
                                # breaks[labels == candidate0[6]] = 255

                candidates = candidates0
                neighborsAreasAll = neighborsAreasAll0

                # seqBreaks.append(breaks)
                # results.append(img)

                salida.write(img)
                print(int(stepbar))
                progressbar.step(stepbar)
                progressbar.after(10, visualizar)
            else:
                cap.release()
                salida.release()
                procesandot.set("LISTO!")

    def abrir_archivo():
        global cap
        global frame_count
        ubicacion.delete(0, END)
        archivo_abierto = filedialog.askopenfilename(
            initialdir="D:/U/PDI/damage_detection-20221115T153600Z-001/damage_detection/data/videos",
            title="Seleccione archivo", filetypes=[
                ("Video", ".mp4"),
                ("Video", ".flv"),
                ("Video", ".avi"),
            ])
        ubicacion.insert(0, archivo_abierto)
        ubicacion.config(state='disabled')
        nombre.config(state='normal')
        cap = cv2.VideoCapture(archivo_abierto)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    procesandot = StringVar()
    cap = None
    frame_count = None
    nom = None
    ubicacion = Entry(fg='grey',font=("Arial", 15))
    ubicacion.place(relx=0.4, rely=0.1, width=500, anchor='c')
    ubicacion.insert(0, "Ubicacion del archivo a procesar")
    ubicacion.bind("<1>", handle_click_ubi)
    Button(text="Cargar video", bg="light blue", command=abrir_archivo,font=("Arial", 15)).place(relx=0.8, rely=0.1, width=200,
                                                                                anchor='c')
    Button(text="Procesar video", bg="light blue", command=procesar,font=("Arial", 15)).place(relx=0.8, rely=0.2, width=200, anchor='c')
    progressbar = ttk.Progressbar()
    progressbar.place(relx=0.5, rely=0.6, width=500, anchor='c')
    procesando = ttk.Label(textvariable=procesandot).place(relx=0.5, rely=0.65, anchor='c')
    nombre = Entry(fg='grey',font=("Arial", 15))
    nombre.place(relx=0.4, rely=0.2, width=500, anchor='c')
    nombre.insert(0, "Ingrese un nombre")
    nombre.config(state='disabled')
    nombre.bind("<1>", handle_click_nom)
    volver = tk.Button(text="Volver", command=abrir_ventana_principal, bg="light blue", font=("Arial", 15))
    volver.place(relx=0.8, rely=0.9, width=200, anchor='c')


def abrir_ventana_cargar_video():
    def visualizar():
        global cap
        global txtbtnpause
        if cap is not None:
            ret, frame = cap.read()
            if ret == True:
                frame = imutils.resize(frame, width=640)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                im = Image.fromarray(frame)
                img2 = ImageTk.PhotoImage(image=im)

                lblVideo.configure(image=img2)
                lblVideo.image = img2
                if (txtbtnpause.get()=="Pause"):
                    lblVideo.after(50, visualizar)
            else:
                # lblInfoVideoPath.configure(text="Aún no se ha seleccionado un video")
                lblVideo.image = ""
                cap.release()

    def reproducir_video():
        global cap
        seleccion = lista.get()
        cargar_video = dir_videos_cargados + '/' + seleccion
        print(cargar_video)
        cap = cv2.VideoCapture(cargar_video)
        botonPause = tk.Button(textvariable=txtbtnpause, command=boton_pause, bg="light blue", font=("Arial", 15)).place(relx=0.2, rely=0.9, width=200, anchor='c')
        visualizar()
    def boton_pause():
        global txtbtnpause
        if(txtbtnpause.get()=="Pause"):
            txtbtnpause.set("Play")
        else:
            txtbtnpause.set("Pause")
            visualizar()

    eliminar_elementos()
    cap = None
    txtbtnpause.set("Pause")
    volver = tk.Button(text="Volver", command=abrir_ventana_principal, bg="light blue", font=("Arial", 15))
    volver.place(relx=0.8, rely=0.9, width=200, anchor='c')
    my_str_var = tk.StringVar()
    my_str_var.set("Seleccione video...")
    lista = ttk.Combobox(font=("Arial", 15), state='readonly',textvariable = my_str_var)
    lista.place(relx=0.4, rely=0.1, width=500, anchor='c')

    opciones = os.listdir(dir_videos_cargados)

    lista['values'] = opciones

    reproducir = Button(text="Reproducir", command=reproducir_video, bg="light blue", font=("Arial", 15))
    reproducir.place(relx=0.8, rely=0.1, width=200, anchor='c')

    lblVideo = Label(root)
    lblVideo.place(relx=0.5, rely=0.5, width=600,height=400,anchor='c')


root = tk.Tk()
root.title("Deteccion de roturas")
root.config(width=900, height=600)
root.resizable(width=False, height=False)
root.title("Detector de Roturas")
width0, height0 = (640, 480)
nNeighbors = 10
salida = None
nAlpha = 6
candidates = []
neighborsAreasAll = []
breakID = 0
img = None
txtbtnpause = StringVar()

bienvenida = tk.Label(root, text="Bienvenidos/as al Detector de Roturas")
bienvenida.config(fg="blue", font=("Arial", 35))
bienvenida.place(relx=0.5, rely=0.3, anchor='c')

nuevoVideo = tk.Button(text="Nuevo video", command=abrir_ventana_nuevo_video, bg="light blue", font=("Arial", 15))
nuevoVideo.place(relx=0.5, rely=0.45, width=200, anchor='c')

cargarVideo = tk.Button(text="Cargar video", command=abrir_ventana_cargar_video, bg="light blue", font=("Arial", 15))
cargarVideo.place(relx=0.5, rely=0.55, width=200, anchor='c')

salir = tk.Button(text="Salir", command=cerrar_ventana, bg="light blue", font=("Arial", 15))
salir.place(relx=0.5, rely=0.65, width=200, anchor='c')

root.mainloop()