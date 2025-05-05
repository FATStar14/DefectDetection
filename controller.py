import serial
import time
import tkinter as tk
from tkinter import messagebox

arduino_port = "COM5"
baud_rate = 9600

try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(5)
except serial.SerialException:
    messagebox.showerror("Error", "Could not connect to Arduino. Check the port.")
    ser = None

root = tk.Tk()
root.title("Defix Detector: Conveyor Control")
root.geometry("400x300")
root.resizable(False, False)


def send_command(command):
    if ser:
        ser.write((command + "\n").encode("utf-8"))

def start_motor():
    send_command("start")
    status_label.config(text="Motor Status: Running", fg="green")

def stop_motor():
    send_command("stop")
    status_label.config(text="Motor Status: Stopped", fg="red")

status_label = tk.Label(root, text="Motor Status: --", font=("Arial", 12))
status_label.pack(pady=10)

start_button = tk.Button(root, text="Start Motor", command=start_motor, bg="green", fg="white", width=15)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Stop Motor", command=stop_motor, bg="red", fg="white", width=15)
stop_button.pack(pady=10)

root.mainloop()
