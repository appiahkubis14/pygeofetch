# Create the folder
mkdir -p ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# Copy the plugin
cp -r ~/Documents/fatal_flaw_platform/qgis_plugin/fatal_flaw_analyzer \
      ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

# Verify
ls ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/fatal_flaw_analyzer/

# Open the report generator
nano ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/fatal_flaw_analyzer/report_generator.py

# Update
cp -r ~/Documents/fatal_flaw_platform/qgis_plugin/fatal_flaw_analyzer \
      ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/


cd ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/

zip -r fatal_flaw_analyzer_v1.0.0.zip fatal_flaw_analyzer/ \
    --exclude "*.pyc" \
    --exclude "*/__pycache__/*" \
    --exclude "*/.git/*"

ls -lh fatal_flaw_analyzer_v1.0.0.zip


cat ~/Documents/fatal_flaw_platform/qgis_plugin/fatal_flaw_analyzer/metadata.txt

https://plugins.qgis.org/plugins/add
name=Fatal Flaw Analyzer
version=1.0.0
qgisMinimumVersion=3.28
author=Your Name
email=your@email.com
description=Fatal Flaw risk assessment for renewable energy sites


# On any web server or S3 bucket, create this structure:
# https://yourserver.com/qgis-plugins/
#   └── plugins.xml
#   └── fatal_flaw_analyzer_v1.0.0.zip


<?xml version="1.0"?>
<plugins>
  <pyqgis_plugin name="Fatal Flaw Analyzer" version="1.0.0">
    <description>Fatal Flaw risk assessment for renewable energy sites</description>
    <version>1.0.0</version>
    <qgis_minimum_version>3.28</qgis_minimum_version>
    <author_name>Your Name</author_name>
    <download_url>https://yourserver.com/qgis-plugins/fatal_flaw_analyzer_v1.0.0.zip</download_url>
    <homepage>https://github.com/yourorg/fatal-flaw-platform</homepage>
    <experimental>False</experimental>
  </pyqgis_plugin>
</plugins>


Settings → Options → Plugins tab
Click Add under Plugin Repositories
Name: Fatal Flaw Platform
URL: https://yourserver.com/qgis-plugins/plugins.xml
Click OK





# Remove the dist directory and build artifacts
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/

# 1. Update version in pyproject.toml
# 2. Build
python -m build

# 3. Upload to TestPyPI
twine upload --repository testpypi dist/*

# 4. Test it
pip install --index-url https://test.pypi.org/simple/ your-package-name

# 5. If everything works, upload to PyPI
twine upload dist/*