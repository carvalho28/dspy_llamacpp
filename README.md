# dspy_llamacpp

**dspy_llamacpp** is a lightweight Python wrapper that seamlessly integrates the `llama-server` from [llama.cpp](https://github.com/ggerganov/llama.cpp) with the [dspy](https://github.com/yourusername/dspy) package. This project abstracts away the complexities of starting, managing, and terminating the llama-server—especially addressing memory management issues encountered with other approaches (such as llama cpp python). With this wrapper, you simply instantiate a class, and everything is set up automatically, letting you focus on using dspy's capabilities.

## Motivation

While there are existing interfaces (like llama cpp python) for interacting with llama.cpp, I encountered numerous memory management issues and complexities in ensuring robust server cleanup. To solve this, I built **dspy_llamacpp**, which:

- Automatically starts the llama-server in its own process group.
- Handles server options in an intuitive, keyword-style manner.
- Ensures the server is gracefully terminated on exit—even in cases of unexpected crashes—using `atexit` and proper signal handling.
- Integrates directly with dspy so that you can use functions like `dspy.predict` without extra boilerplate.

## Features

- **Automatic Server Management:** Instantiates and terminates the llama-server automatically.
- **Custom Server Options:** Easily pass additional options (e.g., `--verbosity`) as keyword-style arguments.
- **Simple Integration with dspy:** Once instantiated, you can call dspy functions directly.
- **Environment-based Configuration:** Use a `.env` file to manage configuration (e.g., the model path).

## Installation

Install the package directly from GitHub via pip:

```bash
pip install git+https://github.com/carvalho28/dspy_llamacpp.git
```

## Usage

### Setting Up the Server

1. **Configure your Environment**:  
Create a `.env` file in your project root with the following content (adjust as needed):
```env
MODEL_PATH=/path/to/your/model
```

2. **Instantiate the Server in Your Code**:  
In your Python script, load the environment variables (using `python-dotenv` if desired) and instantiate the server wrapper. For example:

```python
import os
from dotenv import load_dotenv
from dspy_llamacpp import AutoLlamaCpp

# Load environment variables from the .env file
load_dotenv()

# Retrieve the model path from the environment
MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH:
    raise ValueError("MODEL_PATH is not set in the environment")

# Instantiate the server with custom options (if necessary)
llama_server = AutoLlamaCpp(
    MODEL_PATH,
    port=8080,
    server_options={"--verbosity": -1},
    temperature=0.5,
)
```

3. **Integrate with DSPy**:  
Once the server is running, dspy is automatically configured to use the LM instance pointing to the running server.  
You can now call dspy functions such as dspy.predict directly in your application.

> **Note**: For a complete demonstration, refer to the simple example provided in the examples directory.

## How It Works

**Server Startup:**  
The `AutoLlamaCpp` class starts the `llama-server` in its own process group using `subprocess.Popen` with the provided model path and port. Additional server options can be specified via a dictionary (e.g., `{"--verbosity": -1}`), which are appended to the command line.

**Server Cleanup:**  
The server is automatically terminated when your program exits by leveraging the `atexit` module and signal handling, preventing lingering processes and memory issues.

**dspy Integration:**  
After the server starts, the LM instance is configured to point to the local server. This allows you to seamlessly use dspy’s functionalities (such as `dspy.Predict`) without manual server management.

## Contributing

Contributions are welcome! Please open issues or submit pull requests. For major changes, please open an issue first to discuss ideas.

## Acknowledgements

- [**llama.cpp**](https://github.com/ggml-org/llama.cpp): for the `llama-server`.
- [**dspy**](https://github.com/stanfordnlp/dspy): for the LM framework.
