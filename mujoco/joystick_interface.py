import os
import struct
import threading


class JoystickInterface:
    def __init__(
        self, device_path="/dev/input/js0", max_v_x=1.0, max_v_y=0.5, max_omega=1.0
    ):
        self.device_path = device_path
        self.running = True
        self.available = False

        self.cmd_x = 0.0
        self.cmd_y = 0.0
        self.cmd_yaw = 0.0

        self.MAX_V_X = max_v_x
        self.MAX_V_Y = max_v_y
        self.MAX_OMEGA = max_omega
        self.JOY_MAX = 32767.0

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        if not os.path.exists(self.device_path):
            print(f"[Joystick] device not found: {self.device_path}")
            self.running = False
            return

        print(f"[Joystick] listening on {self.device_path}")
        event_format = "IhBB"
        event_size = struct.calcsize(event_format)

        try:
            with open(self.device_path, "rb") as js_file:
                self.available = True
                while self.running:
                    event_data = js_file.read(event_size)
                    if not event_data:
                        continue

                    _, value, type_evt, number = struct.unpack(event_format, event_data)

                    if type_evt & 0x80:
                        continue

                    if type_evt == 0x02:
                        norm_val = value / self.JOY_MAX
                        if abs(norm_val) < 0.1:
                            norm_val = 0.0

                        if number == 1:
                            self.cmd_x = -norm_val * self.MAX_V_X
                        elif number == 0:
                            self.cmd_y = -norm_val * self.MAX_V_Y
                        elif number == 3:
                            self.cmd_yaw = -norm_val * self.MAX_OMEGA
        except Exception as exc:
            print(f"[Joystick] read error: {exc}")
        finally:
            self.available = False

    def get_command(self):
        return self.cmd_x, self.cmd_y, self.cmd_yaw

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
