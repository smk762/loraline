# Cloudflare Zero Trust

This scaffold can run without exposing the self-LAN API port publicly. Use
`compose.cloudflare.yaml` to publish the service through Cloudflare Tunnel while
keeping the application private inside Docker.

## Goal

- No public inbound listener on the host for the self-lora API
- Cloudflare Tunnel as the internet-facing entrypoint
- Cloudflare Access in front of the tunnel
- Existing bearer-token auth remains enabled on the application for
  defense-in-depth

## Prereqs

- A Cloudflare Zero Trust account
- A tunnel token for the hostname you want to use
- `CLOUDFLARED_TUNNEL_TOKEN` added to `.env`

## Run Through Tunnel Only

This mode disables the normal host port binding by overlaying:

```bash
docker compose -f compose.yaml -f compose.cloudflare.yaml up -d --build
```

In this mode:

- `self-lora` is reachable only inside the compose network
- `cloudflared` is the only external publication path

## Keep LAN + Add Cloudflare

If you want LAN access and Cloudflare at the same time, do not use the overlay
that clears `ports:`. Instead either:

- run just `compose.yaml` and add a separate `cloudflared` service manually, or
- create a second overlay that adds `cloudflared` but does not remove the host port

If you keep LAN access enabled, firewall the LAN port to trusted sources only.

## Access Policy

Recommended Cloudflare Access setup:

- self-hosted application for your chosen hostname
- service-token or identity-based policy for allowed callers
- no bypass/public policy

## App-Side Auth

Cloudflare should not replace service auth. Keep the app bearer token enabled:

```text
Authorization: Bearer <SELF_LORA_INTERNAL_TOKEN>
```

That gives you:

- Cloudflare Access as the edge gate
- bearer auth as the service-level gate

## Firewall Guidance

For tunnel-only mode:

- block inbound `8010` on the host
- allow only SSH/admin access as needed

For LAN + tunnel mode:

- allow `8010` only from the app host, trusted worker IPs, and admin IPs
- deny all other inbound traffic to `8010`
