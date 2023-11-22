import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from model import BiSeNet
import torch
import os
import os.path as osp
import torchvision.transforms as transforms

# Rest of your code remains the same

def similar(G1, B1, R1, G2, B2, R2):
    ar = []
    if G2 > 30:
        ar.append(1000. * G1 / G2)
    if B2 > 30:
        ar.append(1000. * B1 / B2)
    if R2 > 30:
        ar.append(1000. * R1 / R2)
    if len(ar) < 1:
        return False
    if min(ar) == 0:
        return False
    br = max(R1, G1, B1) / max(G2, B2, R2)
    return max(ar) / min(ar) < 1.55 and br > 0.7 and br < 1.4

def CFAR(G, B, R, g, b, r, pro, bri):
    ar = []
    if g > 30:
        ar.append(G / g)
    if b > 30:
        ar.append(B / b)
    if r > 30:
        ar.append(R / r)
    if len(ar) == 0:
        return True
    if bri > 120:
        return max(ar) / min(ar) < 2
    if bri < 70:
        return max(ar) / min(ar) < 1.7
    if pro < 0.35:
        return max(ar) / min(ar) < 1.6 and max(ar) > 0.8
    else:
        return max(ar) / min(ar) < 1.7 and max(ar) > 0.65

def vis_parsing_maps(im, origin, parsing_anno, stride, save_im=False, save_path='output.jpg', mod='gold'):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    num_of_class = np.max(vis_parsing_anno)

    SB = 0
    SR = 0
    SG = 0
    cnt = 0
    total = 0
    brigh = 0
    FB = 0
    FR = 0
    FG = 0
    FN = 0
    GB = 0  # Moved the variable GB to this scope
    GG = 0
    GR = 0
    for x in range(0, origin.shape[0]):
        for y in range(0, origin.shape[1]):
            _x = int(x * 512 / origin.shape[0])
            _y = int(y * 512 / origin.shape[1])
            if vis_parsing_anno[_x][_y] == 1:
                FB = FB + int(origin[x][y][0])
                FG = FG + int(origin[x][y][1])
                FR = FR + int(origin[x][y][2])
                FN = FN + 1
    FB = int(FB / FN)
    FR = int(FR / FN)
    FG = int(FG / FN)

    for x in range(0, origin.shape[0]):
        for y in range(0, origin.shape[1]):
            _x = int(x * 512 / origin.shape[0])
            _y = int(y * 512 / origin.shape[1])
            if vis_parsing_anno[_x][_y] == 17:
                OB = int(origin[x][y][0])
                OG = int(origin[x][y][1])
                OR = int(origin[x][y][2])
                if similar(OB, OG, OR, FB, FG, FR):
                    continue
                SB = SB + OB
                SG = SG + OG
                SR = SR + OR
                cnt = cnt + 1
                brigh = brigh + OR + OG + OR
            if vis_parsing_anno[_x][_y] <= 17:
                total = total + 1
    pro = cnt / total
    SB = int(SB / cnt)
    SG = int(SG / cnt)
    SR = int(SR / cnt)
    brigh = brigh / cnt / 3

    for x in range(0, origin.shape[0]):
        for y in range(0, origin.shape[1]):
            _x = int(x * 512 / origin.shape[0])
            _y = int(y * 512 / origin.shape[1])
            if vis_parsing_anno[_x][_y] == 17:
                OB = int(origin[x][y][0])
                OG = int(origin[x][y][1])
                OR = int(origin[x][y][2])
                if similar(OB, OG, OR, FB, FG, FR):
                    continue
                cur = origin[x][y]
                sum = int(cur[0]) + int(cur[1]) + int(cur[2])
                if mod == 'gold':
                    GB = 0
                    GG = 215 * 0.8
                    GR = 255 * 0.8
                if mod == 'red':
                    GB = 50
                    GG = 80
                    GR = 255
                if mod == 'black':
                    GB = 100
                    GG = 110
                    GR = 125

                if brigh > 120:
                    param = 20
                    p = (sum + param) * (sum + param) / (brigh + param) / (brigh + param) / 20
                elif brigh < 80:
                    p = sum * 70 / 520 / brigh
                else:
                    p = sum / 520
                if CFAR(SB, SG, SR, cur[0], cur[1], cur[2], pro, brigh):
                    cur[0] = min(255, int(GB * p))
                    cur[1] = min(255, int(GG * p))
                    cur[2] = min(255, int(GR * p))

    if save_im:
        # Ensure the save_path has a valid image file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # Add more if needed
        ext = os.path.splitext(save_path)[-1].lower()
        if ext not in valid_extensions:
            save_path += '.jpg'  # Default to JPEG format

        cv2.imwrite(save_path, origin, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

# The rest of your code remains the same

def evaluate(cp='model/model.pth', input_path='4.jpg', output_path='output.jpg', mode='gold'):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cpu()
    save_pth = osp.join('', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(input_path)
        origin = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        image = img.resize((512, 512))
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cpu()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        vis_parsing_maps(image, origin, parsing, stride=1, save_im=True, save_path=output_path, mod=mode)

def change_hair_color(input_image, output_path, color):
    evaluate(input_path=input_image, output_path=output_path, mode=color)

def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg")])
    if file_path:
        entry_image_path.delete(0, tk.END)
        entry_image_path.insert(0, file_path)

def process_image():
    input_image_path = entry_image_path.get()
    output_image_name = entry_output_name.get()
    color = color_var.get()  # Get selected hair color
    output_path = os.path.join("files", output_image_name)
    change_hair_color(input_image_path, output_path, color)
    result_label.config(text=f"Processed image saved as: {output_path}")

# Create the main GUI window
root = tk.Tk()
root.title("Hair Color Change GUI")

# Create a canvas to center the containers
canvas = tk.Canvas(root, width=400, height=400, bg="white")
canvas.pack()

# Create a frame for the entire GUI
frame = tk.Frame(canvas, bg="white")
canvas.create_window((0, 0), window=frame, anchor="nw")

# Create input fields and labels
label_image_path = tk.Label(frame, text="Select an image:", bg="white")
label_image_path.pack()
entry_image_path = tk.Entry(frame)
entry_image_path.pack()
button_browse = tk.Button(frame, text="Browse", command=browse_image)
button_browse.pack()

label_output_name = tk.Label(frame, text="Enter output image name:", bg="white")
label_output_name.pack()
entry_output_name = tk.Entry(frame)
entry_output_name.pack()

# Create radio buttons for hair color selection
label_color_choice = tk.Label(frame, text="Select hair color:", bg="white")
label_color_choice.pack()
color_var = tk.StringVar()
color_var.set("black")  # Default hair color
colors = ["Black", "Blonde", "Brunette", "Red", "Gray"]
color_buttons = []

for color in colors:
    color_buttons.append(tk.Radiobutton(frame, text=color, variable=color_var, value=color.lower()))
    color_buttons[-1].pack()

# Create buttons for image processing
process_button = tk.Button(frame, text="Process Image", command=process_image)
process_button.pack()

# Display the processed image
processed_image_label = tk.Label(frame)
processed_image_label.pack()

result_label = tk.Label(frame, text="")
result_label.pack()

# Start the GUI
root.mainloop()
