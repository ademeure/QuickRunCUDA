import os
import time
import subprocess
from pathlib import Path

class CUDAController:
    def __init__(self):
        self.cmd_pipe_path = "/tmp/quickruncuda_cmd"
        self.resp_pipe_path = "/tmp/quickruncuda_resp"

        # Start QuickRunCUDA in server mode
        self.process = subprocess.Popen(["./QuickRunCUDA", "--server"])

        # Wait for pipes to be created
        while not (os.path.exists(self.cmd_pipe_path) and
                  os.path.exists(self.resp_pipe_path)):
            time.sleep(0.1)

    def send_command(self, args, read_response=True):
        # Convert args to command string
        cmd = " ".join(str(arg) for arg in args)

        # Write to command pipe
        with open(self.cmd_pipe_path, "w") as f:
            f.write(cmd)

        if read_response:
            # Read response
            with open(self.resp_pipe_path, "r") as f:
                response = f.read()

            return response
        else:
            return None

    def __del__(self):
        # Send exit command
        try:
            self.send_command(["exit"], False)
        except:
            pass
        self.process.terminate()
        self.process.wait()