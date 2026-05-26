# ==========================================================
# Sonopet Test Program
# ==========================================================
# This program tests the sonopet_live_data executable.
#    It allows for testing an arduino activated footpedal device
#    It allows for saving the collected data to a json file.
#
# If footpedal port permission denied, use:  sudo usermod -a -G dialout $USER
#
# Commands:
#   sonopet_live_data start
#   sonopet_live_data grab
#   sonopet_live_data stop
#   sonopet_live_data list
#
# Features:
# - Timestamped data grabs
# - Fixed 5 ms interval timing
# - Optional footpedal control
#
# Performance Example:
#   100 grabs in ~0.158 seconds
#   Average ~1.6 ms per grab
# ==========================================================


# ==========================================================
# CONFIGURATION
# ==========================================================

EXE = "/home/btllab/rp-repo/sonopet/Python_SonoDAQ/sonopet_live_data"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 5000

NUM_SAMPLES = 100000
INTERVAL = 0.005  # 5 ms

USE_FOOTPEDAL = True
FOOTPEDAL_PORT = "/dev/ttyACM0"
FOOTPEDAL_BAUD = 115200

USE_FILE_SAVE = True
OUTPUT_DIR_NAME = "run_outputs"


# ==========================================================
# IMPORTS
# ==========================================================

import subprocess
import time
import socket
import json
import os
from datetime import datetime

if USE_FOOTPEDAL:
    import serial


# ==========================================================
# FOOT PEDAL
# ==========================================================

footpedal = None


def build_output_file_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, OUTPUT_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)

    now = datetime.now()
    ns_suffix = f"{time.time_ns() % 10_000_000_000:010d}"
    filename = (
        f"SonopetCase_{now:%Y_%m_%d}_T_{now:%H_%M_%S}_"
        f"{ns_suffix}.json"
    )
    return os.path.join(output_dir, filename)


def setup_footpedal():
    global footpedal

    print("Initializing footpedal...")

    footpedal = serial.Serial(
        FOOTPEDAL_PORT,
        FOOTPEDAL_BAUD,
        timeout=1
    )
    print("Footpedal initialized")
    time.sleep(2)


def send_FP_value(val):

    if footpedal is None:
        return

    footpedal.write(str(val).encode())
    print(f"Sent footpedal value: {val}")


def stop_footpedal():

    if footpedal is None:
        return

    send_FP_value(0)
    time.sleep(1)


def close_footpedal():

    if footpedal:
        footpedal.close()


# ==========================================================
# SERVER
# ==========================================================

def start_server():

    print("Starting server...")

    result = subprocess.run(
        [EXE, "start"],
        check=True,
        capture_output=True,
        text=True
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="")

    if "ERROR:" in result.stdout or "ERROR:" in result.stderr:
        raise RuntimeError(
            "sonopet_live_data failed to start"
        )

    start_wait = time.perf_counter() 
    print("Server connection ip address: ", SERVER_HOST)
    print("Server connection port: ", SERVER_PORT)
    # exit()

    while True:
        try:
            s_test = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )

            s_test.connect(
                (SERVER_HOST, SERVER_PORT)
            )

            s_test.close()

            break

        except ConnectionRefusedError:

            if time.perf_counter() - start_wait > 5:
                raise RuntimeError(
                    "Server failed to start"
                )

            time.sleep(0.1)

    print("Server ready")


def stop_server(sock):

    try:
        sock.sendall(b"stop")
        sock.close()

    except Exception:
        pass


# ==========================================================
# MAIN
# ==========================================================

def main():

    sock = None
    outfile = None

    try:

        # -----------------------------
        # Footpedal
        # -----------------------------

        if USE_FOOTPEDAL:
            setup_footpedal()
            send_FP_value(1)
            time.sleep(5)
            print("Footpedal initialized")

        # -----------------------------
        # File setup
        # -----------------------------

        if USE_FILE_SAVE:
            output_file_path = build_output_file_path()

            outfile = open(
                output_file_path,
                "w",
                buffering=1   # line buffered
            )

            print(
                f"Saving data to {output_file_path}"
            )

        # -----------------------------
        # Start server
        # -----------------------------

        start_server()

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )

        sock.connect(
            (SERVER_HOST, SERVER_PORT)
        )

        start_time = time.perf_counter()
        next_time = start_time

        # -----------------------------
        # Acquisition Loop
        # -----------------------------

        for i in range(NUM_SAMPLES):

            next_time += INTERVAL

            sock.sendall(b"grab")

            data = sock.recv(8192).decode()

            try:

                r = json.loads(data)

                timestamp = time.time()

                record = {
                    "timestamp": timestamp,
                    "data": r
                }

                print(record)

                # Save to file
                if USE_FILE_SAVE:

                    outfile.write(
                        json.dumps(record) + "\n"
                    )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"JSON decode error: {data!r}"
                ) from exc

            # Timing control

            sleep_time = next_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print("WARNING: Loop overrun")

        # -----------------------------
        # Timing results
        # -----------------------------

        end_time = time.perf_counter()

        total_time = end_time - start_time

        print(
            f"Performed {NUM_SAMPLES} grabs "
            f"in {total_time:.3f} seconds"
        )

        print(
            f"Average time per grab: "
            f"{total_time / NUM_SAMPLES:.6f}"
        )

    finally:

        if sock:
            stop_server(sock)

        if outfile:
            outfile.close()
            print("File closed")

        if USE_FOOTPEDAL:
            stop_footpedal()
            close_footpedal()


# ===============================================sudo usermod -a -G dialout $USER===========

if __name__ == "__main__":
    main()
