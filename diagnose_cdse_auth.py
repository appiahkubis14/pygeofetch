import httpx, json

resp = httpx.post(
    "https://m2m.cr.usgs.gov/api/api/json/stable/login-token",
    content=json.dumps({"username": "SamuelYamforo", "token": "pZR!9XSr85145X8fMa7tJkEbs_!esM3Z4iV_EY5FHiuEO1oD@R1eM7PRUOTXNLn1"}),
    headers={"Content-Type": "application/json"},
)
print(resp.json())