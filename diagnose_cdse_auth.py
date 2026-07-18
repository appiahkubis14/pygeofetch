#!/usr/bin/env python3
"""
Standalone USGS M2M authentication diagnostic — bypasses pygeofetch entirely.

Talks directly to the M2M API's login-token endpoint using the exact
official request format, so we can see the RAW response and isolate
whether this is a token problem, a username problem, or an M2M-access-
not-approved problem (three very different, easily confused issues that
all surface as the same generic "AUTH_INVALID" error).
"""
import getpass
import json
import sys

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "httpx", "--quiet"])
    import httpx

BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

print("=" * 70)
print("USGS M2M Authentication Diagnostic")
print("=" * 70)
print()
print("This does NOT use your ERS password — only your username and the")
print("Application Token generated at https://ers.cr.usgs.gov (profile ->")
print("'Application Token').")
print()

username = input("ERS username: ").strip()
token = getpass.getpass("Application Token (hidden): ").strip()

print()
print(f"Username exactly as sent: {username!r}")
print(f"Token length: {len(token)} characters (should be 64)")
print(f"POSTing to: {BASE_URL}/login-token")
print()

payload = {"username": username, "token": token}

try:
    resp = httpx.post(
        f"{BASE_URL}/login-token",
        content=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
except Exception as exc:
    print(f"NETWORK ERROR — could not reach USGS M2M API at all: {exc}")
    sys.exit(1)

print(f"HTTP Status: {resp.status_code}")
print()

try:
    body = resp.json()
    print("Response body:")
    print(json.dumps(body, indent=2))
except Exception:
    print("Raw response (not JSON):")
    print(resp.text[:1000])
    body = {}

print()
print("=" * 70)

error_code = body.get("errorCode")
error_msg = body.get("errorMessage", "")

if resp.status_code == 200 and body.get("data") and not error_code:
    print("SUCCESS — this username + token combination works.")
    print("If pygeofetch still fails with the same credentials, that's a")
    print("pygeofetch-side bug — share this SUCCESS result for further help.")

elif error_code == "AUTH_INVALID":
    print("DIAGNOSIS: AUTH_INVALID — USGS rejected the credentials, but this")
    print("code covers THREE distinct causes. Check each in order:")
    print()
    print("  1. TOKEN LENGTH — you entered", len(token), "characters.")
    if len(token) != 64:
        print("     ⚠ USGS Application Tokens are exactly 64 characters.")
        print("     This token looks WRONG — likely truncated when copied,")
        print("     or this isn't actually the Application Token (e.g. you")
        print("     may have copied part of the page, not the token itself).")
        print("     -> Go to https://ers.cr.usgs.gov -> profile -> 'Application")
        print("        Token' and generate a NEW one, then copy it carefully.")
    else:
        print("     Length looks correct (64 chars) — token format is fine.")
    print()
    print("  2. USERNAME — must be your exact ERS username (not your email,")
    print("     unless your username IS your email). Verify by logging in at")
    print("     https://ers.cr.usgs.gov directly and checking your profile")
    print("     for the exact username string.")
    print()
    print("  3. M2M ACCESS NOT APPROVED — this is the MOST COMMON cause.")
    print("     Having a valid Application Token does NOT automatically grant")
    print("     M2M API access — that's a SEPARATE manual approval USGS staff")
    print("     review individually. Check your approval status:")
    print("     -> https://ers.cr.usgs.gov/profile/access")
    print("     If it shows 'pending' or you never submitted a request, submit")
    print("     one there and wait for approval (not instant — can take days).")

elif error_code:
    print(f"DIAGNOSIS: Unrecognized errorCode {error_code!r} — {error_msg}")
    print("Share this exact errorCode and errorMessage for further diagnosis.")

else:
    print(f"DIAGNOSIS: Unexpected response — share the full body above.")