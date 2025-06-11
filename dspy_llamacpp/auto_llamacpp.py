import subprocess
import time
import atexit
import os
import signal
import dspy


class AutoLlamaCpp:
    def __init__(
        self,
        model_path,
        port=8080,
        sleep_time=10,
        is_reasoning=False,
        is_json=False,
        server_options: dict | None = None,
        gpu_env: str | None = None,
        **lm_kwargs,
    ):
        """
        model_path: Path to the model.
        port: Port to run the server on.
        server_options: Dictionary of additional server options, e.g. {"--verbosity": -1}
        gpu_env: Comma-separated list of GPU indices to expose (e.g., "1,2,3").
        lm_kwargs: Additional keyword arguments for dspy.LM.
        """
        self.model_path = model_path
        self.port = port
        self.sleep_time = sleep_time
        self.server_options = server_options or {}
        self.gpu_env = gpu_env
        self.proc = None
        self._start_server()

        # Configure the LM instance for DSPy
        self.lm = dspy.LM(
            model="openai/model",
            api_base=f"http://localhost:{port}/v1",
            api_key="none",
            **lm_kwargs,
        )
        # dspy.configure(lm=self.lm)
        if is_reasoning:
            print("Using reasoning adapter")
            adapter = dspy.TwoStepAdapter(self.lm)
            dspy.configure(lm=self.lm, adapter=adapter)
        elif is_json:
            print("Using JSON adapter")
            adapter = dspy.JSONAdapter()
            dspy.configure(lm=self.lm, adapter=adapter)
        else:
            dspy.configure(lm=self.lm)

    def _start_server(self):
        # Base command for llama-server
        cmd = [
            "llama-server",
            "-m",
            self.model_path,
            "--port",
            str(self.port),
        ]
        # Add server options to the command
        for key, value in self.server_options.items():
            if isinstance(value, bool) and value:  # Only include flag if True
                cmd.append(key)
            elif value is None:  # For flags passed as None
                cmd.append(key)
            elif not isinstance(value, bool):
                cmd.extend([key, str(value)])

        # Prepare environment with optional GPU masking
        env = os.environ.copy()
        if self.gpu_env:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_env
            print(f"[AutoLlamaCpp] Using CUDA_VISIBLE_DEVICES={self.gpu_env}")

        # Start the server subprocess
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid, env=env)
        print("Llama server started on port", self.port)

        # Wait for server to spin up
        time.sleep(self.sleep_time)

        # Ensure it shuts down cleanly on script exit
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
