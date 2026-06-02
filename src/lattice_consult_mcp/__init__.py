"""lattice-consult-mcp -- cross-provider LLM consultation MCP server.

First concrete impl of Trellis DSM-26 AngularTechnique substrate per FP-MSG-310 §4.

Side-effect on import: `truststore.inject_into_ssl()` patches Python's `ssl`
module to use the operating system's certificate store instead of the bundled
`certifi` CA bundle. This is required on systems with corporate / antivirus root
CA injection (e.g., Avast, common Windows + endpoint-security setups) where the
OS store has the necessary intermediate cert but `certifi` does not, causing
SSL handshakes to fail with CERTIFICATE_VERIFY_FAILED. The inject covers all
downstream HTTP libs (httpx, urllib, the OpenAI/Anthropic/xAI SDKs) in one
shot. Soft-fails on platforms where truststore is unavailable.
"""
__version__ = "0.1.0"

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:  # pragma: no cover -- truststore missing -> use certifi fallback
    pass
