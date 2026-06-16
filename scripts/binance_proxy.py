"""scripts/binance_proxy.py — Minimal read-only Binance proxy.

Binance answers requests from some datacenter regions (e.g. the US region where
Hugging Face Spaces run) with HTTP 451. Run this proxy on a host in a Binance-
ALLOWED region (your home PC / a Raspberry Pi / a free VM such as Oracle Cloud
Always-Free in Mumbai). The Space then points its Binance base URLs here, so the
actual calls originate from the allowed IP.

Routes by the first path segment:
    /eapi/...     -> https://eapi.binance.com      (options chain)
    /fapi/...     -> https://fapi.binance.com       (perp premium/OI)
    /futures/...  -> https://fapi.binance.com       (long/short, taker ratios)
    /api/...      -> https://api.binance.com         (spot klines)

Run it:
    pip install fastapi "uvicorn[standard]" httpx
    set PROXY_SECRET=some-long-secret           # optional but recommended
    uvicorn binance_proxy:app --host 0.0.0.0 --port 8080
    # (run from the scripts/ dir, or `uvicorn scripts.binance_proxy:app ...`)

Then on the HF Space set these secrets (all three point at the same proxy):
    BINANCE_EAPI         = https://<your-proxy-host>
    BINANCE_FAPI         = https://<your-proxy-host>
    BINANCE_SPOT         = https://<your-proxy-host>
    BINANCE_PROXY_SECRET = some-long-secret       # must match PROXY_SECRET here

Only GETs to the four prefixes above are forwarded; everything else is rejected,
and the optional shared secret keeps the open proxy from being abused.
"""
from __future__ import annotations

import os

import httpx
from fastapi import FastAPI, Request, Response

HOSTS = {
    "eapi": "https://eapi.binance.com",
    "fapi": "https://fapi.binance.com",
    "futures": "https://fapi.binance.com",
    "api": "https://api.binance.com",
}
SECRET = os.getenv("PROXY_SECRET", "")

app = FastAPI(title="Binance read proxy")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/{full_path:path}")
async def proxy(full_path: str, request: Request):
    if SECRET and request.headers.get("x-proxy-secret") != SECRET:
        return Response("forbidden", status_code=403)
    prefix = full_path.split("/", 1)[0]
    base = HOSTS.get(prefix)
    if not base:
        return Response("unknown route", status_code=404)
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(f"{base}/{full_path}",
                                 params=dict(request.query_params))
    except Exception as e:                                   # noqa: BLE001
        return Response(f"upstream error: {e}", status_code=502)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))
