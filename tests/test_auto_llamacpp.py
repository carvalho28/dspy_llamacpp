import unittest
from unittest.mock import patch, MagicMock
import os
import signal

from dspy_llamacpp import AutoLlamaCpp

class TestAutoLlamaCpp(unittest.TestCase):
    @patch('dspy_llamacpp.auto_llamacpp.subprocess.Popen')
    @patch('dspy_llamacpp.auto_llamacpp.time.sleep', return_value=None)
    @patch('dspy_llamacpp.auto_llamacpp.atexit.register')
    def test_server_startup(self, mock_atexit, mock_sleep, mock_popen):
        # Setup a fake process to be returned by subprocess.Popen
        fake_proc = MagicMock()
        fake_proc.pid = 1234
        mock_popen.return_value = fake_proc

        # Instantiate your AutoLlamaCpp, which should call _start_server()
        auto_llama = AutoLlamaCpp("dummy_model", port=8080)

        # Verify that Popen was called with the expected command
        expected_cmd = [
            "llama-server",
            "-m", "dummy_model",
            "--port", "8080",
        ]
        mock_popen.assert_called_with(
            expected_cmd,
            preexec_fn=os.setsid
        )

        # Verify that sleep was called to simulate waiting for the server
        mock_sleep.assert_called_once()

        # Check that atexit.register was called to register the cleanup function
        mock_atexit.assert_called_once()

        # Verify that the LM instance is set up (assuming dspy.LM produces something non-None)
        self.assertIsNotNone(auto_llama.lm)

    @patch('dspy_llamacpp.auto_llamacpp.os.killpg')
    @patch('dspy_llamacpp.auto_llamacpp.os.getpgid', return_value=1234)
    def test_cleanup(self, mock_getpgid, mock_killpg):
        # Create a fake process
        fake_proc = MagicMock()
        fake_proc.pid = 1234
        # Instantiate without starting the actual server by assigning our fake process
        auto_llama = AutoLlamaCpp.__new__(AutoLlamaCpp)
        auto_llama.proc = fake_proc

        # Call the cleanup method
        auto_llama._cleanup()

        # Check that os.killpg was called correctly
        mock_getpgid.assert_called_with(fake_proc.pid)
        mock_killpg.assert_called_with(1234, signal.SIGTERM)
        # Ensure that wait() was called on the process to finish cleanup
        fake_proc.wait.assert_called_once()

if __name__ == '__main__':
    unittest.main()
