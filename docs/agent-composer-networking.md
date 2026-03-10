# Agent-Composer Networking Notes

These notes describe the safest update path for `agent-composer` when you want:

- internal-only services like Ollama to stay private
- LAN access for selected app endpoints
- host/IP allowlisting for internal services such as `self-lora`

## Recommended Split

- Keep `ollama` on the internal Docker network only
- Keep `rag-chat` as the user-facing/API-facing service
- Keep `self-lora` as a separate internal/LAN service unless you later merge it into the same compose stack
- Expose only the services that actually need LAN access

## Whitelisted LAN Access

For services that should be reachable from the LAN but not from everywhere:

1. Publish the service port on the host
2. Add a host firewall allowlist
3. Keep bearer auth enabled at the app layer

Example policy for a self-lora host:

- allow `8010/tcp` from the `agent-composer` host IP
- allow `8010/tcp` from your admin workstation IP
- deny all other sources

This is best enforced with `ufw`, `nftables`, or your router firewall, not only with Docker.

## Internal-Only Ollama

The current `ollama` setup in `agent-composer` is correct for security:

- no host port published
- healthcheck against `127.0.0.1:11434` inside the container
- attached only to the internal network

Do not publish raw Ollama to the LAN unless you add a proxy in front of it.

## If Self-Lora Needs Chat Access Later

Safer options, in order:

1. Let `agent-composer` continue owning chat and call its app endpoints instead of Ollama directly
2. Add a small authenticated proxy in front of Ollama
3. Only if both stacks are on the same Docker host, join `self-lora` to the same internal network and use `http://ollama:11434`

## Reverse Proxy Pattern For LAN

If `agent-composer` is exposed on LAN, put a reverse proxy in front and allowlist source IPs there too.

Good pattern:

- `rag-chat` behind Caddy/Nginx/Traefik
- TLS for internet-facing access
- LAN allowlist for internal/private routes
- public routes exposed only where necessary

## Suggested Agent-Composer Updates

- Keep `ollama` internal-only
- Keep `rag-ingest` internal-only unless an external uploader truly needs it
- Publish only `rag-chat` or a reverse proxy in front of it
- Add host firewall rules for allowed source IPs
- Add a separate internal service config for calling `self-lora` over LAN:

```text
LORA_API_BASE_URL=http://<self-lora-lan-ip>:8010
LORA_API_AUTH_HEADER=Authorization
LORA_API_AUTH_SCHEME=Bearer
LORA_API_KEY=<shared-internal-token>
```

## If You Later Merge Self-Lora Into Agent-Composer

If you move this scaffold into `agent-composer`, prefer:

- keep it as its own internal service
- keep its API boundary intact
- let it join the same internal network as Ollama and any future image/video workers

That preserves the current separation while avoiding cross-host LAN traffic.
