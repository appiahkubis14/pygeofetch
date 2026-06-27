# Screenshots

Place CLI screenshot images here so they render in the GitHub README.

## How to capture and add them

### 1. Install a terminal screenshot tool

```bash
# Option A: Use terminalizer (records and exports as GIF or PNG)
npm install -g terminalizer

# Option B: Just take a screenshot with your desktop tool
# Ubuntu: gnome-screenshot, Flameshot, or Shift+PrtScr
# macOS: Cmd+Shift+4
```

### 2. Capture the search demo

```bash
# Run this command and screenshot the terminal output
pygeofetch search run \
    --bbox "-74.1,40.6,-73.7,40.9" \
    --start-date 2024-01-01 \
    --end-date 2024-03-01 \
    --cloud-cover 0-20 \
    --providers aws_earth,planetary_computer \
    --format table
```

Save the screenshot as: `docs/assets/search_demo.png`

### 3. Capture the download demo

```bash
# Run this and screenshot the progress bar output
pygeofetch download run \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 2 \
    --max-items 3
```

Save the screenshot as: `docs/assets/download_demo.png`

### 4. Commit and push

```bash
git add docs/assets/search_demo.png docs/assets/download_demo.png
git commit -m "docs: add CLI screenshot demos"
git push
```

The images will then display in the README on GitHub.

## Image naming convention

| File | Used in README |
|---|---|
| `search_demo.png` | Quick Start → Search section |
| `download_demo.png` | Quick Start → Download section |
| `status_demo.png` | Optional: status command |
| `providers_demo.png` | Optional: providers list |
