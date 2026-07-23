"""
Standalone OpenTopography authentication test — bypasses pygeofetch
entirely, so you can see exactly what OpenTopography's own server says,
not just the bare 401 that gets wrapped into a generic AuthenticationError.

Usage:
    python test_opentopography_auth.py YOUR_API_KEY
"""

import sys
import httpx

def test_key(api_key: str) -> None:
    if not api_key or api_key.strip() in ("your_api_key", ""):
        print("!! You're passing the placeholder value, not a real key.")
        print("   Get a real key from https://portal.opentopography.org")
        print("   (Sign in -> My Account -> Request API Key)")
        return

    print(f"Testing key: {api_key[:6]}...{api_key[-4:] if len(api_key) > 10 else ''}")
    print(f"Key length: {len(api_key)} characters\n")

    # Smallest possible real request: a 0.01-degree box, cheapest DEM type
    # (SRTMGL3, 90m) -- if auth is the problem, it'll fail identically to
    # a larger request, but faster and without wasting quota either way.
    url = (
        "https://portal.opentopography.org/API/globaldem"
        "?demtype=SRTMGL3"
        "&south=5.60&north=5.61&west=-0.20&east=-0.19"
        "&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )

    print("Sending a direct request to OpenTopography's real API...\n")
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            resp = client.get(url)
    except httpx.RequestError as exc:
        print(f"!! Network-level failure before any response: {exc}")
        print("   Check your internet connection / firewall, not your API key.")
        return

    print(f"HTTP status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type', 'unknown')}")
    print(f"Response size: {len(resp.content)} bytes\n")

    if resp.status_code == 200:
        if resp.headers.get("content-type", "").startswith("image"):
            print("SUCCESS — this is a real, valid, working API key.")
            print("If pygeofetch is still failing with this same key, the")
            print("bug is in pygeofetch specifically, not the key itself —")
            print("report back with this result, since that changes the")
            print("diagnosis completely.")
        else:
            # 200 OK but not actually image data usually means an error
            # message returned WITH a 200 status, which does happen with
            # some OpenTopography failure modes
            print("Got 200 OK but the response isn't image data.")
            print("Response body (first 500 chars):")
            print(resp.text[:500])
    elif resp.status_code == 401:
        print("401 Unauthorized — OpenTopography rejected this key outright.")
        print("Response body (often has a specific reason):")
        print(resp.text[:500])
        print("\nMost likely causes, in order of probability:")
        print("  1. Key was just created and needs a short activation delay")
        print("  2. Key was copy-pasted with extra whitespace or a typo")
        print("  3. Key was revoked or the account's email wasn't verified")
    elif resp.status_code == 403:
        print("403 Forbidden — key is recognized but not permitted for this request.")
        print("Response body:")
        print(resp.text[:500])
    elif resp.status_code == 429:
        print("429 Too Many Requests — rate limited. Wait and retry, not a key problem.")
    else:
        print(f"Unexpected status {resp.status_code}. Response body:")
        print(resp.text[:500])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_opentopography_auth.py YOUR_API_KEY")
        sys.exit(1)
    test_key(sys.argv[1])