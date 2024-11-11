import cv2
from tkinter import Tk, messagebox, Button, Label, StringVar, Entry, PhotoImage, Text, Canvas
from PIL import ImageTk
import json
from main import cnn
from pathlib import Path

#全局配置
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r".\images\assets\frame0")

#全局圖像變量，防止被GC回收
button_image_1, entry_image_1, entry_image_2 = [None]*3


# 全局變量
image = None
label1 = None
isordinary = None
iscnn = None

def relative_to_assets(path: str) -> Path:
    """獲得全路徑"""
    return ASSETS_PATH / Path(path)

def load_images()
    """載入圖片"""
    global button_image_1, entry_image_1, entry_image_2
    button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
    entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
    entry_image_2 = PhotoImage(file=relative_to_assets("entry_2.png"))

def onmouse(event, x, y, flags, param):
    global buttoncolor, istake, name, label1, isordinary, iscnn, state
    """鼠标狀態
  
    Args:
        event(cv2.EVENT): 鼠標事件
        x(float): 鼠標所在位置的橫坐標
        y(float): 鼠標所在位置的縱坐標
    """
    # 當點擊鼠標左鍵時
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 < x < 140 and 10 < y < 60:
            cv2.imwrite(filename = r"user_headshot/{}.png".format(name.get()), img = image)  # 寫入文件

            # 檢驗平凡，靚仔，還是靚女
            if not iscnn:#控制答案
                c = cnn(r"{}.png".format(name.get()), [10.2])
            else:
                c = iscnn
            if c == "one":
                state="靚仔"
            elif c == "two":
                state = "平凡"
            elif c == "three":
                state = "靚女"
            else:
                state = "醜"

            isordinary.set(state)

            # 寫入json文件
            with open("data.json", 'r') as f:
                json_data = json.load(f)
                json_data[name.get()] = c
            with open("data.json", 'w') as fw:
                json.dump(json_data,fw)

            #更新圖片
            photo_new = ImageTk.PhotoImage(file=r"./user_headshot/{}.png".format(name.get()))
            label1.config(image=photo_new)
            label1.image = photo_new

            messagebox.showinfo("提示", "成功拍攝相片！")
            istake = False #拍完照
    # 當鼠標移動時
    elif event == cv2.EVENT_MOUSEMOVE:
        if 10 < x < 140 and 10 < y < 60:
            buttoncolor = (0, 0, 255)  # 改變按鈕顏色
        else:
            buttoncolor = (0, 255, 0)  # 恢復按鈕顏色


def capture():
    """
    拍照
    """
    global image, istake, label1, iscnn
    istake = True #是否拍完照
    win.withdraw()
    cv2.namedWindow("photo")
    cv2.setMouseCallback("photo", onmouse)
    iscnn = 0 #是否需要測試(0為需要)

    # 循環
    while cap.isOpened() and istake:#拍完照退出循環
        b, image = cap.read()
        if not b:
            break

        # 創建按鈕
        cv2.rectangle(image, (10, 10), (140, 60), buttoncolor, -1)
        cv2.putText(image, "Take a Photo", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 顯示圖片
        cv2.imshow("photo", image)

        # 按下了q就退出
        if cv2.waitKey(1) == ord("q"):
            win.deiconify()
            break
        #避免出事
        if cv2.waitKey(1) == ord("g"):
            iscnn = "three"
            print("user_headshot/{}.png\t{}".format(name.get(), iscnn))
        if cv2.waitKey(1) == ord("b"):
            iscnn = "one"
            print("user_headshot/{}.png\t{}".format(name.get(), iscnn))
        if cv2.waitKey(1) == ord("o"):
            iscnn = "two"
            print("user_headshot/{}.png\t{}".format(name.get(), iscnn))
        if cv2.waitKey(1) == ord("n"):
            iscnn = 0
            print("cancel")
        if cv2.waitKey(1) == ord("y"):
            iscnn = "four"
            print("user_headshot/{}.png\t{}".format(name.get(), iscnn))
    cv2.destroyAllWindows()
    win.deiconify()


# =======================================
#   主程式
# =======================================
if __name__ == "__main__":
    win = Tk()
    win.geometry("1280x814")
    win.configure(bg="#FFFFFF")
    win.title("ordinary or not")
    win.iconbitmap(r"images/camera.ico")
    photo = ImageTk.PhotoImage(file=r"images/welcome.jpg")
    load_images()

    try:#如果沒有内置攝像頭，就用外置攝像頭
        cap = cv2.VideoCapture(0)
    except:
        cap = cv2.VideoCapture(1)
    buttoncolor = (0, 255, 0)

    #介面和插件設定
    canvas = Canvas(
        win,
        bg="#FFFFFF",
        height=814,
        width=1280,
        bd=0,
        highlightthickness=0,
        relief="ridge"
    )

    canvas.place(x=0, y=0)
    canvas.create_rectangle(
        0.0,
        0.0,
        1280.0,
        153.0,
        fill="#1F5ACA",
        outline="")
    canvas.create_text(
        535.0,
        17.0,
        anchor="nw",
        text="你的樣貌是：",
        fill="#FFFFFF",
        font=("Inter", 36 * -1)
    )
    canvas.create_text(
        344.0,
        707.0,
        anchor="nw",
        text="請輸入你的姓名：",
        fill="#000000",
        font=("Inter", 36 * -1)
    )

    canvas.create_text(
        514.0,
        676.0,
        anchor="nw",
        text="已為你和1408名勞校學生配對",
        fill="#000000",
        font=("Inter", 18 * -1)
    )
    #是否平凡
    isordinary = StringVar()
    isordinary.set("")
    label_bg_1 = canvas.create_image(
        639.5,
        107.0,
        image=entry_image_2
    )
    label3 = Label(
        bd=0,
        bg="#D9D9D9",
        fg="#000716",
        highlightthickness=0,
        textvariable=isordinary,
        font=("Inter", 40 * -1)
    )
    label3.place(
        x=375.0,
        y=82.0,
        width=529.0,
        height=48.0
    )

    #名字
    name = StringVar()
    name.set("Please enter your name(no chinese)")

    #匹配
    button_1 = Button(
        image=button_image_1,
        borderwidth=0,
        highlightthickness=0,
        command=capture,
        relief="flat")
    button_1.place(
        x=542.0,
        y=756.0,
        width=195.0,
        height=44.0
    )

    #名字輸入框
    entry_bg_1 = canvas.create_image(
        790.5,
        728.5,
        image=entry_image_1
    )
    entry1 = Entry(
        bd=0,
        bg="#D9D9D9",
        fg="#000716",
        highlightthickness=0,
        textvariable=name,
        font=("Inter", 20 * -1)
    )
    entry1.place(
        x=653.5,
        y=715.0,
        width=274.0,
        height=25.0
    )
    entry1.focus_set()

    #照片
    label1 = Label(win, image=photo)
    label1.place(x=320.0, y=186.0)


    win.resizable(False, False)
    win.mainloop()
    cap.release()
