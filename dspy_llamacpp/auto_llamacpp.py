import subprocess
import time
import atexit
import os
import signal
import dspy

class AutoLlamaCpp:
    def __init__(self, model_path, port=8080, sleep_time=10, server_options=None, **lm_kwargs):
        """
        model_path: Path to the model.
        port: Port to run the server on.
        server_options: Dictionary of additional server options, e.g. {"--verbosity": -1}
        lm_kwargs: Additional keyword arguments for dspy.LM.
        """
        self.model_path = model_path
        self.port = port
        self.sleep_time = sleep_time
        self.server_options = server_options or {}
        self.proc = None
        self._start_server()
        # Configure the LM instance
        self.lm = dspy.LM(
            model="openai/model",
            api_base=f"http://localhost:{port}/v1",
            api_key="none",
            **lm_kwargs,
        )
        # remove for thread safety ??
        # dspy.configure(lm=self.lm)

    def _start_server(self):
        # Base command for llama-server
        cmd = [
            "llama-server",
            "-m", self.model_path,
            "--port", str(self.port),
        ]
        # Append server options from the dictionary
        for key, value in self.server_options.items():
            # Append the key (option flag)
            cmd.append(key)
            # If value is not a boolean, append its value
            if not isinstance(value, bool):
                cmd.append(str(value))
        # Start the server in its own process group
        self.proc = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid
        )
        print("Llama server started on port", self.port)
        # Wait for the server to be available
        time.sleep(self.sleep_time)
        # Register cleanup to ensure termination on exit
        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.proc:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception as e:
                print("Error terminating llama server:", e)
            self.proc.wait()

    def close(self):
        """Public method to manually stop the server."""
        self._cleanup()
