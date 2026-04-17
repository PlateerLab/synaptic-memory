"""Third-party framework adapters (LangChain, LlamaIndex, ...).

Each adapter is lazy-imported — importing ``synaptic.integrations``
by itself does not require any of the third-party dependencies.
Install the specific integration you need::

    pip install "synaptic-memory[langchain]"
    pip install "synaptic-memory[llamaindex]"  # planned
"""
