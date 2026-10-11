# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub private vulnerability
reporting:

https://github.com/anywave/lattice-consult-mcp/security/advisories/new

Do not open a public issue for a suspected vulnerability. We aim to acknowledge
reports within 7 days.

## Scope notes

- This server never stores API keys; it reads them from environment variables.
  Do not commit a `.env` file or paste keys into issues.
- `privacy_tier: "local_only"` limits dispatch to the configured Jan endpoint.
  It is a routing filter, not encryption: the prompt goes wherever
  `LATTICE_JAN_BASE_URL` points.
