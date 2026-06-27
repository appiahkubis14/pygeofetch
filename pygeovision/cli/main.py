"""
PyGeoVision CLI — production-ready geospatial AI command line interface.

Integrates:
  pygeofetch — satellite data (search, download, auth, pipeline, cache)
  geoai      — AI inference (segment, detect, classify, train)

Usage:
    pygeovision --help
    pygeovision status
    pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06
    pygeovision data download --from-search results.geojson --output ./data/
    pygeovision data auth add usgs --username USER --password PASS
    pygeovision data auth add planet --api-key YOUR_KEY
    pygeovision data pipeline run weekly.yaml
    pygeovision ai segment buildings --input scene.tif --output buildings.tif
    pygeovision ai detect cars --input aerial.tif --output cars.geojson
    pygeovision ai train segmentation --data ./chips/ --output model.pth
    pygeovision pipeline building_footprints --bbox ... --output ./results
    pygeovision models list
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(package_name="pygeovision")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """PyGeoVision — Geospatial AI Platform (pygeofetch + geoai)."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# =========================================================================
# DATA commands — delegates to pygeofetch CLI
# =========================================================================

@cli.group()
def data() -> None:
    """Satellite data commands — powered by pygeofetch.

    \b
    All 22 providers: planetary_computer, copernicus, usgs, nasa_earthdata,
    planet, sentinel_hub, maxar_gbdx, airbus_oneatlas, aws_earth, element84,
    noaa_big_data, alaska_satellite_facility, opentopography, google_earth_engine,
    terrabotics, esa_scihub, jaxa_earth, isro_bhuvan, inpe_cbers, digitalglobe,
    nasa_earthdata_cloud, geoserver_generic
    """


@data.group()
def auth() -> None:
    """Manage satellite provider credentials (stored in system keyring)."""


@auth.command("add")
@click.argument("provider")
@click.option("--username", help="Username (usgs, nasa_earthdata).")
@click.option("--password", help="Password (usgs, nasa_earthdata).")
@click.option("--api-key", help="API key (planet, opentopography, airbus_oneatlas).")
@click.option("--client-id", help="OAuth2 client ID (copernicus, sentinel_hub, maxar_gbdx).")
@click.option("--client-secret", help="OAuth2 client secret.")
def auth_add(provider, username, password, api_key, client_id, client_secret):
    """Add credentials for a satellite data provider.

    \b
    Examples:
        pygeovision data auth add usgs --username USER --password PASS
        pygeovision data auth add planet --api-key PL_KEY
        pygeovision data auth add copernicus --client-id ID --client-secret SECRET
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.add_credentials(
        provider,
        username=username, password=password,
        api_key=api_key, client_id=client_id, client_secret=client_secret,
    )
    click.echo(f"✓ Credentials added for '{provider}'")


@auth.command("list")
def auth_list():
    """List providers with stored credentials."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    providers = client.data.list_credentials()
    if not providers:
        click.echo("No credentials stored.")
    else:
        click.echo("Stored credentials for:")
        for p in providers:
            click.echo(f"  • {p}")


@auth.command("test")
@click.argument("provider")
def auth_test(provider):
    """Test connectivity to a provider."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.test_provider(provider)
    if ok:
        click.echo(f"✓ {provider}: reachable")
    else:
        click.echo(f"✗ {provider}: unreachable or not authenticated", err=True)
        sys.exit(1)


@auth.command("remove")
@click.argument("provider")
@click.option("--yes", is_flag=True, help="Skip confirmation.")
def auth_remove(provider, yes):
    """Remove stored credentials for a provider."""
    if not yes:
        click.confirm(f"Remove credentials for '{provider}'?", abort=True)
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.data.remove_credentials(provider)
    click.echo(f"✓ Credentials removed for '{provider}'")


@data.command("search")
@click.option("--bbox", nargs=4, type=float, required=True,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT",
              help="Bounding box in WGS84.")
@click.option("--date", "date_str", default=None,
              help="Date (YYYY-MM) or range (YYYY-MM-DD/YYYY-MM-DD).")
@click.option("--start-date", default=None, help="Start date YYYY-MM-DD.")
@click.option("--end-date", default=None, help="End date YYYY-MM-DD.")
@click.option("--providers", default=None,
              help="Comma-separated provider IDs (e.g. planetary_computer,copernicus).")
@click.option("--satellite", default=None,
              help="Satellite shortcut: sentinel-2, landsat, planet, dem, etc.")
@click.option("--collections", default=None,
              help="STAC collection IDs (e.g. sentinel-2-l2a,landsat-c2-l2).")
@click.option("--cloud-max", default=30.0, show_default=True, help="Max cloud cover %%.")
@click.option("--max-results", default=50, show_default=True)
@click.option("--sort-by", default="datetime",
              type=click.Choice(["datetime", "cloud_cover", "score", "satellite"]))
@click.option("--output", "-o", default=None, help="Save results to GeoJSON file.")
@click.option("--format", "fmt", default="table",
              type=click.Choice(["table", "json", "ids"]))
def data_search(bbox, date_str, start_date, end_date, providers, satellite,
                collections, cloud_max, max_results, sort_by, output, fmt):
    """Search satellite imagery across 22+ pygeofetch providers.

    \b
    Examples:
        # Open access — no credentials needed
        pygeovision data search --bbox -0.15 51.47 -0.10 51.52 \\
            --date 2024-06 --collections sentinel-2-l2a --cloud-max 10

        # Multi-provider with credentials
        pygeovision data search --bbox -74.1 40.6 -73.7 40.9 \\
            --start-date 2024-01-01 --end-date 2024-06-01 \\
            --providers planetary_computer,copernicus,usgs --cloud-max 20

        # Satellite shortcut
        pygeovision data search --bbox -74.1 40.6 -73.7 40.9 \\
            --date 2024-06 --satellite sentinel-2
    """
    from pygeovision import PyGeoVision

    # Resolve date range
    if date_str:
        if "/" in date_str:
            start, end = date_str.split("/", 1)
        elif len(date_str) == 7:  # YYYY-MM
            start = f"{date_str}-01"
            end = f"{date_str}-28"
        else:
            start = end = date_str
    else:
        start = start_date or "2024-01-01"
        end = end_date or "2024-12-31"

    prov_list = [p.strip() for p in providers.split(",")] if providers else None
    coll_list = [c.strip() for c in collections.split(",")] if collections else None

    client = PyGeoVision()
    click.echo(f"Searching pygeofetch providers... (bbox={tuple(bbox)}, dates={start}→{end})")

    results = client.search(
        bbox=tuple(bbox),
        date_range=(start, end),
        providers=prov_list,
        satellite=satellite,
        collections=coll_list,
        cloud_cover_max=cloud_max,
        max_results=max_results,
        sort_by=sort_by,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} scenes:\n")

    if fmt == "json":
        data_out = json.dumps([r.to_dict() for r in results], indent=2, default=str)
        click.echo(data_out)
        if output:
            Path(output).write_text(data_out)
    elif fmt == "ids":
        for r in results:
            click.echo(r.id)
        if output:
            Path(output).write_text("\n".join(r.id for r in results))
    else:
        # Table format
        click.echo(f"{'ID':<40} {'Provider':<22} {'Date':<12} {'Cloud':>6} {'Score':>6} {'Satellite'}")
        click.echo("-" * 110)
        for r in results:
            cc = f"{r.cloud_cover:.0f}%" if r.cloud_cover is not None else "N/A"
            score = f"{r.score:.2f}" if r.score else "N/A"
            click.echo(f"{r.id[:38]:<40} {r.provider:<22} {r.date:<12} {cc:>6} {score:>6} {r.satellite}")

        if output:
            # Save as GeoJSON
            features = []
            for r in results:
                features.append({
                    "type": "Feature",
                    "id": r.id,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [r.bbox[0], r.bbox[1]], [r.bbox[2], r.bbox[1]],
                            [r.bbox[2], r.bbox[3]], [r.bbox[0], r.bbox[3]],
                            [r.bbox[0], r.bbox[1]],
                        ]] if r.bbox else None,
                    },
                    "properties": r.to_dict(),
                })
            geojson = {"type": "FeatureCollection", "features": features}
            Path(output).write_text(json.dumps(geojson, indent=2, default=str))
            click.echo(f"\nSaved {len(results)} results → {output}")


@data.command("download")
@click.option("--from-search", "search_file", type=click.Path(exists=True),
              help="GeoJSON results file from 'data search'.")
@click.option("--bbox", nargs=4, type=float, default=None,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT",
              help="Download all results in this bbox (requires --date).")
@click.option("--date", "date_str", default=None)
@click.option("--providers", default=None)
@click.option("--output", "-o", default="./downloads", show_default=True)
@click.option("--parallel", default=4, show_default=True)
@click.option("--verify-checksum", is_flag=True, default=False)
@click.option("--resume", is_flag=True, default=True)
@click.option("--retry", default=5, show_default=True)
@click.option("--post-process", default=None,
              help="Post-processing chain: unzip,reproject:EPSG:4326,compress:lzw,cog")
@click.option("--bandwidth-limit", default=None, help="Download throttle in MB/s.")
@click.option("--on-failure", default="skip",
              type=click.Choice(["skip", "abort", "retry"]))
@click.option("--max-items", default=None, type=int)
def data_download(search_file, bbox, date_str, providers, output, parallel,
                  verify_checksum, resume, retry, post_process, bandwidth_limit,
                  on_failure, max_items):
    """Download satellite scenes via pygeofetch.

    \b
    Post-process actions (chained):
        unzip                    Extract ZIP/TAR archives
        reproject:EPSG:4326      Reproject to CRS
        compress:lzw             Apply compression (lzw/deflate/zstd)
        ndvi                     Compute NDVI band
        ndwi                     Compute NDWI band
        cog                      Cloud Optimized GeoTIFF
        resample:10              Resample to N metres
        clip:area.geojson        Clip to geometry
        merge                    Merge overlapping scenes
        pan-sharpen              Pan-sharpen multispectral
        atmospheric:sen2cor      Atmospheric correction

    \b
    Examples:
        pygeovision data download --from-search results.geojson \\
            --output ./data/ --parallel 4 \\
            --post-process unzip,reproject:EPSG:4326,compress:lzw,cog

        pygeovision data download --from-search results.geojson \\
            --output ./data/ --verify-checksum
    """
    from pygeovision import PyGeoVision
    from pygeovision.data.fetch import SatelliteFetcher

    client = PyGeoVision()
    items = []

    if search_file:
        # Parse GeoJSON from search command
        fetcher = SatelliteFetcher()
        items = fetcher._parse_stac_geojson_file(Path(search_file))
    elif bbox and date_str:
        start = f"{date_str}-01" if len(date_str) == 7 else date_str
        end = f"{date_str}-28" if len(date_str) == 7 else date_str
        prov_list = [p.strip() for p in providers.split(",")] if providers else None
        items = client.search(
            bbox=tuple(bbox), date_range=(start, end), providers=prov_list
        )
    else:
        click.echo("Provide --from-search or (--bbox + --date)", err=True)
        sys.exit(1)

    if not items:
        click.echo("No items to download.")
        return

    if max_items:
        items = items[:max_items]

    pp = [p.strip() for p in post_process.split(",")] if post_process else None

    click.echo(f"Downloading {len(items)} scenes → {output}...")
    results = client.download(
        items,
        output_dir=output,
        parallel=parallel,
        verify_checksum=verify_checksum,
        resume=resume,
        retry_attempts=retry,
        post_process=pp,
        bandwidth_limit_mb=float(bandwidth_limit) if bandwidth_limit else None,
        on_failure=on_failure,
    )

    ok = sum(1 for r in results if r.success)
    fail = len(results) - ok
    total_mb = sum(r.size_mb for r in results if r.success)

    click.echo(f"\n✓ {ok} succeeded ({total_mb:.1f} MB total)")
    if fail:
        click.echo(f"✗ {fail} failed", err=True)
    for r in results:
        click.echo(f"  {r}")


@data.group()
def pipeline() -> None:
    """Manage and run pygeofetch YAML data pipelines."""


@pipeline.command("run")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option("--step", default=None, help="Run only this named step.")
def pipeline_run(yaml_file, step):
    """Run a pygeofetch YAML pipeline.

    \b
    Example pipeline (weekly-sentinel2.yaml):
        name: weekly-sentinel2-ndvi
        schedule: "0 6 * * 1"
        steps:
          - search:
              providers: [planetary_computer, copernicus]
              date_range: last_7_days
              cloud_cover: 0-10
              bbox: "-74.1,40.6,-73.7,40.9"
          - filter:
              expression: "data.cloud_cover < 5"
          - download:
              parallel: 4
              output: ./raw/
              verify_checksum: true
          - export:
              format: cloud_optimized_geotiff
              destination: s3://my-bucket/
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    click.echo(f"Running pipeline: {yaml_file}")
    result = client.run_pipeline_yaml(yaml_file, step=step)
    if result.get("success"):
        click.echo("✓ Pipeline completed successfully.")
    else:
        click.echo(f"✗ Pipeline failed: {result.get('stderr', '')[:300]}", err=True)
        sys.exit(1)


@pipeline.command("validate")
@click.argument("yaml_file", type=click.Path(exists=True))
def pipeline_validate(yaml_file):
    """Validate a pygeofetch pipeline YAML without running it."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.data.validate_pipeline(yaml_file)
    if ok:
        click.echo(f"✓ Pipeline valid: {yaml_file}")
    else:
        click.echo("✗ Pipeline invalid", err=True)
        sys.exit(1)


@pipeline.command("schedule")
@click.argument("yaml_file", type=click.Path(exists=True))
@click.option("--name", default=None, help="Schedule name.")
@click.option("--cron", default=None, help="Cron expression (e.g. '0 6 * * 1').")
def pipeline_schedule(yaml_file, name, cron):
    """Schedule a pygeofetch pipeline for recurring execution."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ok = client.data.schedule_pipeline(yaml_file, name=name, cron=cron)
    if ok:
        click.echo(f"✓ Pipeline scheduled: {name or yaml_file}")
    else:
        click.echo("✗ Scheduling failed (pygeofetch CLI required)", err=True)


@pipeline.command("list")
def pipeline_list():
    """List all scheduled pygeofetch pipelines."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    pipelines = client.data.list_scheduled_pipelines()
    if not pipelines:
        click.echo("No scheduled pipelines.")
    else:
        for p in pipelines:
            click.echo(f"  {p.get('name', 'unnamed')}: {p.get('schedule', 'no schedule')}")


@pipeline.command("history")
@click.option("--limit", default=20, show_default=True)
def pipeline_history(limit):
    """Show pygeofetch pipeline run history."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    history = client.data.pipeline_history(limit=limit)
    if not history:
        click.echo("No pipeline history.")
    else:
        for run in history:
            click.echo(f"  {run}")


@data.group()
def cache() -> None:
    """Manage pygeofetch search result cache."""


@cache.command("stats")
def cache_stats():
    """Show cache statistics."""
    from pygeovision import PyGeoVision
    stats = PyGeoVision().cache_stats()
    click.echo(f"Entries: {stats.get('entries', 0)}")
    click.echo(f"Size:    {stats.get('size_mb', 0):.1f} MB")
    click.echo(f"Location: {stats.get('location', 'N/A')}")


@cache.command("clear")
@click.option("--provider", default=None, help="Clear only this provider.")
@click.option("--older-than", default=None, help="Clear entries older than (e.g. 7d, 1h).")
@click.option("--dry-run", is_flag=True)
def cache_clear(provider, older_than, dry_run):
    """Clear the pygeofetch search result cache."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.data.clear_cache(provider=provider, older_than=older_than, dry_run=dry_run)
    if not dry_run:
        click.echo("✓ Cache cleared.")
    else:
        click.echo("(dry-run) Would clear cache.")


@data.command("providers")
@click.option("--open-only", is_flag=True, help="Only open-access providers.")
@click.option("--sar", is_flag=True, help="Only SAR-capable providers.")
@click.option("--sub-meter", is_flag=True, help="Only sub-metre resolution providers.")
def data_providers(open_only, sar, sub_meter):
    """List all 22 pygeofetch satellite data providers."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    caps = []
    if sar: caps.append("sar")
    if sub_meter: caps.append("sub_meter")
    providers = client.list_providers(open_only=open_only, capabilities=caps or None)
    click.echo(f"\n{'ID':<30} {'Name':<30} {'Auth':>8} {'SAR':>5} {'<1m':>5} {'STAC':>5}")
    click.echo("-" * 90)
    for pid, info in sorted(providers.items()):
        auth = "🔐" if info.get("auth") else "🌐"
        sar_mark = "✓" if info.get("sar") else ""
        sub_mark = "✓" if info.get("sub_meter") else ""
        stac_mark = "✓" if info.get("stac") else ""
        click.echo(f"{pid:<30} {info['name']:<30} {auth:>8} {sar_mark:>5} {sub_mark:>5} {stac_mark:>5}")
    click.echo(f"\nTotal: {len(providers)} providers")


# =========================================================================
# AI commands — powered by geoai
# =========================================================================

@cli.group()
def ai() -> None:
    """Geospatial AI commands — powered by geoai.

    \b
    Capabilities:
        segment  — Building footprints, solar panels, water, agriculture, SAM
        detect   — Cars, ships, parking, grounded SAM, RF-DETR
        classify — Scene classification, land cover, CLIP zero-shot
        change   — ChangeSTAR bi-temporal change detection
        train    — Train segmentation, detection, classification models
        infer    — Tiled inference on large GeoTIFF scenes
        embed    — Satellite image embeddings
        cloud    — Cloud mask generation
    """


@ai.command("segment")
@click.argument("target", type=click.Choice([
    "buildings", "solar", "water", "agriculture", "custom"
]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output mask GeoTIFF.")
@click.option("--vector", default=None, help="Output GeoJSON/Shapefile for polygons.")
@click.option("--model", default=None, help="Model name or HuggingFace Hub ID.")
@click.option("--confidence", default=0.5, show_default=True)
def ai_segment(target, input_path, output, vector, model, confidence):
    """Segment geospatial features using geoai.

    \b
    Examples:
        pygeovision ai segment buildings --input sentinel2.tif \\
            --output buildings.tif --vector buildings.geojson

        pygeovision ai segment water --input s2.tif \\
            --output water.tif --band-order sentinel2

        pygeovision ai segment solar --input aerial.tif \\
            --output solar.tif --vector solar.geojson
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    click.echo(f"Running geoai {target} segmentation on {input_path}...")

    method = getattr(ga.segment, target, None) if target != "custom" else ga.segment.custom
    if target == "buildings":
        result = ga.segment.buildings(input_path, output_path=output,
                                      output_vector=vector, confidence_threshold=confidence)
    elif target == "solar":
        result = ga.segment.solar_panels(input_path, output_path=output, output_vector=vector)
    elif target == "water":
        result = ga.segment.water(input_path, output_path=output)
    elif target == "agriculture":
        result = ga.segment.agriculture_fields(input_path, output_path=output, output_vector=vector)
    elif target == "custom" and model:
        result = ga.segment.custom(input_path, model, output_path=output)
    else:
        click.echo("--model required for custom segmentation", err=True)
        sys.exit(1)

    click.echo(f"✓ Segmentation complete → {output}")
    if vector:
        click.echo(f"✓ Vectors saved → {vector}")


@ai.command("detect")
@click.argument("target", type=click.Choice([
    "cars", "ships", "parking", "grounded", "rfdetr", "multiclass"
]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output GeoJSON detections.")
@click.option("--prompt", default=None, help="Text prompt for grounded SAM detection.")
@click.option("--model", default=None, help="Model path or Hub ID.")
def ai_detect(target, input_path, output, prompt, model):
    """Detect objects in satellite/aerial imagery using geoai.

    \b
    Examples:
        pygeovision ai detect cars --input aerial.tif --output cars.geojson
        pygeovision ai detect ships --input port.tif --output ships.geojson
        pygeovision ai detect grounded --input aerial.tif \\
            --output out.geojson --prompt "swimming pools"
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    click.echo(f"Running geoai {target} detection on {input_path}...")

    if target == "cars":
        ga.detect.cars(input_path, output_path=output)
    elif target == "ships":
        ga.detect.ships(input_path, output_path=output)
    elif target == "parking":
        ga.detect.parking(input_path, output_path=output)
    elif target == "grounded":
        if not prompt:
            click.echo("--prompt required for grounded detection", err=True)
            sys.exit(1)
        ga.detect.grounded(input_path, prompt, output_path=output)
    elif target == "rfdetr":
        ga.detect.rfdetr(input_path, model_id=model, output_path=output)
    elif target == "multiclass":
        if not model:
            click.echo("--model required for multiclass detection", err=True)
            sys.exit(1)
        ga.detect.multiclass(input_path, model, output_path=output)

    click.echo(f"✓ Detection complete → {output}")


@ai.command("classify")
@click.argument("mode", type=click.Choice(["scene", "land-cover", "batch"]))
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--model", default=None, help="Model path or Hub ID.")
@click.option("--classes", default=None, help="Comma-separated class names (land-cover).")
def ai_classify(mode, input_path, model, classes):
    """Classify satellite imagery using geoai.

    \b
    Examples:
        pygeovision ai classify scene --input tile.tif --model classifier.pth
        pygeovision ai classify land-cover --input s2.tif \\
            --classes "forest,water,urban,agriculture"
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    if mode == "scene":
        if not model:
            click.echo("--model required for scene classification", err=True)
            sys.exit(1)
        result = ga.classify.classify(input_path, model)
        click.echo(f"Classification: {result}")
    elif mode == "land-cover":
        class_list = [c.strip() for c in classes.split(",")] if classes else ["forest", "water", "urban", "agriculture"]
        result = ga.classify.land_cover(input_path, classes=class_list)
        click.echo(f"Land cover classification complete.")
    elif mode == "batch":
        if not model:
            click.echo("--model required for batch classification", err=True)
            sys.exit(1)
        ga.classify.batch(input_path, model)
        click.echo("Batch classification complete.")


@ai.command("train")
@click.argument("task", type=click.Choice([
    "segmentation", "detection", "classification", "land-cover", "instance"
]))
@click.option("--data", "-d", "data_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output model checkpoint path.")
@click.option("--val-data", default=None, type=click.Path(exists=True))
@click.option("--num-classes", default=2, show_default=True)
@click.option("--epochs", default=50, show_default=True)
@click.option("--batch-size", default=8, show_default=True)
@click.option("--backbone", default="resnet50", show_default=True)
@click.option("--loss-fn", default="dice",
              type=click.Choice(["dice", "focal", "tversky", "unified_focal", "cross_entropy"]))
def ai_train(task, data_dir, output, val_data, num_classes, epochs, batch_size,
             backbone, loss_fn):
    """Train geospatial AI models using geoai.

    \b
    Examples:
        # Train a building segmentation model
        pygeovision ai train segmentation --data ./building_chips/ \\
            --output building_model.pth --num-classes 2 --epochs 100

        # Train a land cover model with unified focal loss
        pygeovision ai train land-cover --data ./lc_chips/ \\
            --output lc_model.pth --num-classes 11 --loss-fn unified_focal

        # Train an object detector
        pygeovision ai train detection --data ./nwpu_chips/ \\
            --output detector.pth --num-classes 10
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    click.echo(f"Training geoai {task} model...")
    click.echo(f"  Data: {data_dir} | Classes: {num_classes} | Epochs: {epochs}")

    if task == "segmentation":
        ga.train.segmentation(data_dir, output, val_data=val_data,
                              num_classes=num_classes, epochs=epochs,
                              batch_size=batch_size, backbone=backbone)
    elif task == "land-cover":
        ga.train.segmentation_landcover(data_dir, output, num_classes=num_classes,
                                        loss_fn=loss_fn)
    elif task == "detection":
        ga.train.detection(data_dir, output, val_data=val_data,
                           num_classes=num_classes, epochs=epochs)
    elif task == "classification":
        ga.train.classifier(data_dir, output, num_classes=num_classes,
                            backbone=backbone, epochs=epochs)
    elif task == "instance":
        ga.train.instance_segmentation(data_dir, output, num_classes=num_classes,
                                       epochs=epochs)

    click.echo(f"✓ Model trained → {output}")


@ai.command("infer")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--model", "-m", required=True, help="Model path or registry name.")
@click.option("--output", "-o", required=True, help="Output prediction GeoTIFF.")
@click.option("--num-classes", default=2, show_default=True)
@click.option("--tile-size", default=512, show_default=True)
@click.option("--overlap", default=64, show_default=True)
def ai_infer(input_path, model, output, num_classes, tile_size, overlap):
    """Run tiled inference on a large GeoTIFF using geoai.

    \b
    Examples:
        # Using geoai with a custom model
        pygeovision ai infer --input large_scene.tif \\
            --model custom_model.pth --output prediction.tif --num-classes 5

        # Using PyGeoVision's model registry
        pygeovision ai infer --input scene.tif \\
            --model unet_resnet50 --output pred.tif
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    click.echo(f"Running tiled inference: {input_path} → {output}")

    # Try geoai semantic_segmentation first (most common)
    try:
        ga.infer.semantic_segmentation(input_path, model, output_path=output)
        click.echo(f"✓ Inference complete → {output}")
    except Exception:
        # Fall back to PyGeoVision's own inference
        from pygeovision.ai.models import ModelHub
        from pygeovision.ai.inference import TiledInference
        hub = ModelHub()
        m = hub.load(model, num_classes=num_classes)
        engine = TiledInference(m, tile_size=tile_size, overlap=overlap)
        engine.run(input_path, output, num_classes=num_classes)
        click.echo(f"✓ Inference complete → {output}")


@ai.command("chips")
@click.option("--image", required=True, type=click.Path(exists=True), help="Source imagery GeoTIFF.")
@click.option("--label", required=True, type=click.Path(exists=True), help="Label/mask GeoTIFF.")
@click.option("--output", "-o", required=True, help="Output chip directory.")
@click.option("--chip-size", default=256, show_default=True)
@click.option("--overlap", default=0, show_default=True)
def ai_chips(image, label, output, chip_size, overlap):
    """Export training chips from imagery + labels using geoai.

    \b
    Example:
        pygeovision ai chips --image sentinel2.tif --label buildings.tif \\
            --output ./training_chips/ --chip-size 256
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    ga = client.geoai

    click.echo(f"Exporting {chip_size}×{chip_size} chips → {output}...")
    ga.train.generate_chips(image, label, output, chip_size=chip_size, overlap=overlap)
    click.echo(f"✓ Chips exported → {output}")


@ai.command("cloud-mask")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output cloud mask GeoTIFF.")
def ai_cloud_mask(input_path, output):
    """Generate cloud mask using geoai.

    \b
    Example:
        pygeovision ai cloud-mask --input sentinel2.tif --output cloud.tif
    """
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    client.geoai.cloud.predict(input_path, output_path=output)
    click.echo(f"✓ Cloud mask → {output}")


# =========================================================================
# End-to-end PIPELINE commands (pygeofetch data + geoai)
# =========================================================================

@cli.command("pipeline")
@click.argument("pipeline_name", type=click.Choice([
    "change_detection", "land_cover", "building_footprints",
    "crop_monitoring", "disaster_assessment", "deforestation",
    "urban_growth", "water_bodies", "solar_detection", "carbon_estimation",
    "list",
]))
@click.option("--bbox", nargs=4, type=float, default=None,
              metavar="MIN_LON MIN_LAT MAX_LON MAX_LAT")
@click.option("--output", "-o", default="./pipeline_output", show_default=True)
@click.option("--date", default="2024-06", help="YYYY-MM acquisition date.")
@click.option("--date-before", default=None, help="Before date (bi-temporal pipelines).")
@click.option("--date-after", default=None, help="After date (bi-temporal pipelines).")
@click.option("--model", default=None, help="Override default AI model.")
@click.option("--source", default=None, help="Data source override (e.g. worldcover).")
def pipeline_cmd(pipeline_name, bbox, output, date, date_before, date_after, model, source):
    """Run a complete end-to-end geospatial AI pipeline.

    Downloads satellite data via pygeofetch then applies geoai AI models.

    \b
    Available pipelines:
        building_footprints    Segment buildings (geoai + pygeofetch)
        land_cover             Land cover classification (ESA WorldCover / geoai)
        change_detection       Bi-temporal change detection (geoai)
        water_bodies           Surface water mapping (pygeofetch NDWI + geoai)
        solar_detection        Solar panel detection (geoai)
        crop_monitoring        Crop type mapping (geoai)
        disaster_assessment    Rapid damage assessment (geoai)
        deforestation          Forest loss detection (geoai)
        urban_growth           Urban expansion monitoring (geoai)
        carbon_estimation      Biomass/carbon via NDVI (pygeofetch + geoai)
        list                   Show all available pipelines

    \b
    Examples:
        pygeovision pipeline building_footprints \\
            --bbox -0.15 51.47 -0.10 51.52 --date 2024-06

        pygeovision pipeline change_detection \\
            --bbox -74.1 40.6 -73.7 40.9 \\
            --date-before 2020-01 --date-after 2024-01

        pygeovision pipeline water_bodies \\
            --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --source ndwi
    """
    from pygeovision.ai.pipelines import list_pipelines

    if pipeline_name == "list":
        click.echo("\nAvailable PyGeoVision end-to-end pipelines:\n")
        for name in list_pipelines():
            click.echo(f"  {name}")
        return

    if not bbox:
        click.echo("--bbox required", err=True)
        sys.exit(1)

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    kwargs = {"date": date}
    if date_before: kwargs["date_before"] = date_before
    if date_after: kwargs["date_after"] = date_after
    if model: kwargs["model"] = model
    if source: kwargs["source"] = source

    click.echo(f"Running pipeline '{pipeline_name}'...")
    click.echo(f"  bbox={tuple(bbox)} | date={date} | output={output}")

    result = client.pipeline(pipeline_name, bbox=tuple(bbox), output_dir=output, **kwargs)

    if result.success:
        click.echo(f"\n✓ Pipeline complete!")
        if result.output_path:
            click.echo(f"  Output: {result.output_path}")
        if result.stats:
            click.echo("  Stats:")
            for k, v in result.stats.items():
                fmt_v = f"{v:.4f}" if isinstance(v, float) else v
                click.echo(f"    {k}: {fmt_v}")
    else:
        click.echo(f"\n✗ Pipeline failed: {result.error}", err=True)
        sys.exit(1)


# =========================================================================
# MODELS commands
# =========================================================================

@cli.group()
def models() -> None:
    """AI model management commands."""


@models.command("list")
@click.option("--task", default=None,
              type=click.Choice(["segmentation", "detection", "classification",
                                 "change_detection", "super_resolution"]))
@click.option("--pretrained-only", is_flag=True)
def models_list(task, pretrained_only):
    """List all registered PyGeoVision AI models (14+ built-in)."""
    from pygeovision.ai.models.registry import registry
    results = registry.list_models(task=task, pretrained_only=pretrained_only)
    if not results:
        click.echo("No models found.")
        return
    click.echo(f"\n{'Name':<35} {'Task':<22} {'Architecture':<25} {'Pretrained'}")
    click.echo("-" * 90)
    for m in results:
        pt = "✓" if m.pretrained_available else ""
        click.echo(f"{m.name:<35} {m.task:<22} {m.architecture:<25} {pt}")
    click.echo(f"\nTotal: {len(results)} models")


@models.command("info")
@click.argument("model_name")
def models_info(model_name):
    """Show details for a specific model."""
    from pygeovision.ai.models.registry import registry
    try:
        info = registry.get(model_name)
        click.echo(f"\nModel: {info.name}")
        click.echo(f"  Task:        {info.task}")
        click.echo(f"  Architecture:{info.architecture}")
        click.echo(f"  Input bands: {info.input_bands}")
        click.echo(f"  Pretrained:  {'Yes' if info.pretrained_available else 'No'}")
        click.echo(f"  Description: {info.description}")
        if info.tags:
            click.echo(f"  Tags:        {', '.join(info.tags)}")
        if info.paper_url:
            click.echo(f"  Paper:       {info.paper_url}")
    except KeyError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@models.command("cache")
@click.option("--clear", is_flag=True)
@click.option("--model", default=None)
def models_cache(clear, model):
    """Manage AI model checkpoint cache."""
    from pygeovision.ai.models import ModelHub
    hub = ModelHub()
    if clear:
        hub.clear_cache(model_name=model)
        click.echo(f"✓ Cache cleared{f' for {model}' if model else ''}.")
    else:
        cached = hub.list_cached()
        if not cached:
            click.echo("No models cached.")
        else:
            click.echo(f"{'Name':<35} {'Size':<10} {'Path'}")
            for m in cached:
                size = f"{m.local_path.stat().st_size / 1024 / 1024:.1f}MB" if m.local_path.exists() else "?"
                click.echo(f"{m.name:<35} {size:<10} {m.local_path}")


# =========================================================================
# STATUS command
# =========================================================================

@cli.command("status")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status_cmd(as_json):
    """Show PyGeoVision, pygeofetch, and geoai system status."""
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    status = client.status()

    if as_json:
        click.echo(json.dumps(status, indent=2, default=str))
        return

    click.echo("\n══════════════════════════════════════════")
    click.echo("         PyGeoVision System Status        ")
    click.echo("══════════════════════════════════════════")
    click.echo(f"PyGeoVision v{status['pygeovision_version']} | Python {status['python']}")
    click.echo()

    pf = status["pygeofetch"]
    pf_icon = "✓" if pf["available"] else "✗"
    click.echo(f"pygeofetch  {pf_icon}  v{pf['version']} | {pf['providers']} providers | {pf['open_providers']} open-access")

    ga = status["geoai"]
    ga_icon = "✓" if ga["available"] else "✗"
    ga_ver = ga.get("version", "not installed")
    click.echo(f"geoai       {ga_icon}  {ga_ver}")

    torch = status["torch"]
    if torch.get("available") is False:
        click.echo("torch       ✗  not installed")
    else:
        cuda = f" | CUDA {torch.get('gpu', 'N/A')}" if torch.get("cuda") else ""
        click.echo(f"torch       ✓  v{torch.get('version', '?')} | device={torch.get('device', 'cpu')}{cuda}")

    rio = status.get("rasterio")
    click.echo(f"rasterio    {'✓' if rio else '✗'}  {rio or 'not installed'}")

    gpd = status.get("geopandas")
    click.echo(f"geopandas   {'✓' if gpd else '✗'}  {gpd or 'not installed'}")

    click.echo(f"\nRegistered AI models: {status.get('registered_ai_models', 0)}")
    click.echo()

    if not pf["available"]:
        click.echo("⚠  pygeofetch CLI not found in PATH.")
        click.echo("   Run: pip install pygeofetch  (installs PyGeoFetch CLI)")
    if not ga["available"]:
        click.echo("⚠  geoai not installed.")
        click.echo("   Run: pip install geoai-py")


@cli.command("doctor")
def doctor_cmd():
    """Diagnose PyGeoVision installation and all component connectivity."""
    from pygeovision import PyGeoVision
    try:
        client = PyGeoVision()
        result = client.doctor()
    except Exception as exc:
        click.echo(f"\nCould not initialise PyGeoVision: {exc}", err=True)
        return

    click.echo("\n══════════════════════════════════════════")
    click.echo("         PyGeoVision Diagnostics          ")
    click.echo("══════════════════════════════════════════")
    for key, val in result.items():
        if key == "summary":
            continue
        if isinstance(val, dict):
            ok_icon = "✓" if val.get("ok") else "✗"
            extra = ""
            if val.get("version"):
                extra = f"  v{val['version']}"
            if val.get("cuda"):
                extra += f"  CUDA ({val.get('gpu', 'GPU')})"
            if val.get("error"):
                extra = f"  ERROR: {val['error']}"
            click.echo(f"  {ok_icon}  {key:<18}{extra}")
        else:
            click.echo(f"     {key:<18}: {val}")
    click.echo()
    summary = result.get("summary", "Diagnosis complete.")
    click.echo(f"  → {summary}")
    click.echo()

    # Actionable advice
    for key, val in result.items():
        if isinstance(val, dict) and not val.get("ok") and val.get("error"):
            click.echo(f"  ⚠  {key}: {val['error']}")



# ═══════════════════════════════════════════════════════════════════════════════
# NEW CLI COMMAND GROUPS — Phase 6
# ═══════════════════════════════════════════════════════════════════════════════

# ── models ────────────────────────────────────────────────────────────────────
@cli.group("models")
def models_grp():
    """Model registry commands — list, info, download architectures."""

@models_grp.command("list")
@click.option("--task", "-t", default=None, help="Filter by task (segmentation|detection|...)")
@click.option("--family", "-f", default=None, help="Filter by family (resnet|vit|unet|...)")
@click.option("--max-params", type=float, default=None, help="Maximum parameters (millions)")
def models_list(task, family, max_params):
    """List available model architectures."""
    from pygeovision.models.registry import model_registry
    names = model_registry.list(task=task, family=family, max_params_m=max_params)
    click.echo(f"\n  Found {len(names)} models:\n")
    click.echo(f"  {'Name':<28} {'Task':<18} {'Family':<14} {'Params M':>8}  Description")
    click.echo("  " + "-"*90)
    for name in sorted(names):
        spec = model_registry[name]
        click.echo(f"  {name:<28} {spec.task:<18} {spec.family:<14} {spec.params_m:>8.1f}  {spec.description[:35]}")
    click.echo()

@models_grp.command("info")
@click.argument("name")
def models_info(name):
    """Show detailed info for a model."""
    from pygeovision.models.registry import model_registry
    try:
        spec = model_registry[name]
    except KeyError as e:
        click.echo(f"Error: {e}", err=True); return
    click.echo(f"\n  Model: {spec.name}")
    click.echo(f"  Task:  {spec.task}")
    click.echo(f"  Family:{spec.family}")
    click.echo(f"  Params:{spec.params_m}M")
    if spec.hf_id:  click.echo(f"  HF ID: {spec.hf_id}")
    if spec.timm_id:click.echo(f"  timm:  {spec.timm_id}")
    click.echo(f"  Desc:  {spec.description}")
    click.echo(f"  Pretrained on: {spec.pretrained_on}")
    click.echo()

@models_grp.command("download")
@click.argument("name")
@click.option("--cache-dir", default=None, help="Local cache directory")
def models_download(name, cache_dir):
    """Download pretrained weights for a model."""
    from pygeovision.models.registry import model_registry
    from pygeovision.models.weights.downloader import WeightDownloader
    try:
        spec = model_registry[name]
    except KeyError as e:
        click.echo(f"Error: {e}", err=True); return
    if not spec.hf_id and not spec.timm_id:
        click.echo(f"No downloadable weights for {name}"); return
    downloader = WeightDownloader(cache_dir=cache_dir)
    if spec.hf_id:
        click.echo(f"Downloading {name} from HuggingFace Hub...")
        path = downloader.download(spec.hf_id)
        click.echo(f"✓ Downloaded: {path}")
    else:
        click.echo(f"  {name} uses timm — loaded automatically on first use.")

@models_grp.command("summary")
def models_summary():
    """Show registry summary by task."""
    from pygeovision.models.registry import model_registry
    s = model_registry.summary()
    click.echo(f"\n  Model Registry: {s['total']} architectures\n")
    for task, n in sorted(s["by_task"].items()):
        click.echo(f"  {task:<25} {n:>3} models")
    click.echo(f"\n  HF Hub weights:  {s['with_hf_weights']}")
    click.echo(f"  timm weights:    {s['with_timm_weights']}\n")


# ── infer ─────────────────────────────────────────────────────────────────────
@cli.group("infer")
def infer_grp():
    """Run AI inference on satellite imagery."""

@infer_grp.command("predict")
@click.argument("image_path")
@click.option("--model", "-m", default="unet-r50", show_default=True)
@click.option("--output", "-o", default=None)
@click.option("--classes", "-c", type=int, default=2, show_default=True)
@click.option("--chip-size", default=512, show_default=True)
@click.option("--overlap", default=64, show_default=True)
@click.option("--blend", default="gaussian", show_default=True,
               type=click.Choice(["gaussian","linear","constant"]))
@click.option("--device", default=None, help="cuda|cpu|mps")
def infer_predict(image_path, model, output, classes, chip_size, overlap, blend, device):
    """Run tiled inference on a GeoTIFF."""
    import pathlib
    output = output or str(pathlib.Path(image_path).with_suffix("")) + f"_{model}_pred.tif"
    click.echo(f"\n  Inference: {image_path}")
    click.echo(f"  Model:     {model} | classes={classes} | chip={chip_size} | blend={blend}")
    try:
        from pygeovision.models.registry import get_model
        m = get_model(model, num_classes=classes)
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=m, chip_size=chip_size, overlap=overlap,
                              blend_mode=blend, num_classes=classes, device=device)
        result = inf.infer(image_path, output)
        click.echo(f"  ✓ Output:  {output}")
        click.echo(f"  ✓ Chips:   {result.get('n_chips',0)} | Time: {result.get('duration_seconds',0)}s\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)

@infer_grp.command("batch")
@click.argument("input_dir")
@click.argument("output_dir")
@click.option("--model", "-m", default="unet-r50")
@click.option("--workers", "-w", default=2, show_default=True)
@click.option("--pattern", default="*.tif", show_default=True)
def infer_batch(input_dir, output_dir, model, workers, pattern):
    """Run batch inference on a directory of GeoTIFFs."""
    click.echo(f"\n  Batch inference: {input_dir} → {output_dir}")
    try:
        from pygeovision.models.registry import get_model
        from pygeovision.inference.batch import BatchInferenceEngine
        m = get_model(model)
        engine = BatchInferenceEngine(model=m, n_workers=workers)
        result = engine.run_directory(input_dir, output_dir, pattern=pattern)
        click.echo(f"  ✓ {result.get('n_success',0)}/{result.get('n_success',0)+result.get('n_failed',0)} succeeded")
        click.echo(f"  ✓ Time: {result.get('total_time_s',0)}s | {result.get('throughput_fps',0)} fps\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)


# ── label ─────────────────────────────────────────────────────────────────────
@cli.group("label")
def label_grp():
    """Auto-labeling from OSM, satellite data, and foundation models."""

@label_grp.command("osm")
@click.argument("bbox", nargs=4, type=float)
@click.option("--output", "-o", default="./labels/osm.tif")
@click.option("--categories", "-c", multiple=True,
               default=["buildings","roads","water"], show_default=True)
def label_osm(bbox, output, categories):
    """Generate labels from OpenStreetMap."""
    from pygeovision.labeling.osm import OSMLabeler
    click.echo(f"\n  OSM Labeling: bbox={tuple(bbox)}")
    result = OSMLabeler().label(bbox, categories=list(categories), output_path=output)
    if result.get("success"):
        click.echo(f"  ✓ {result['n_features']} features → {output}")
    else:
        click.echo(f"  ✗ {result.get('error')}", err=True)

@label_grp.command("quality")
@click.argument("label_path")
@click.option("--html", is_flag=True, help="Save HTML report")
def label_quality(label_path, html):
    """Assess label quality."""
    from pygeovision.labeling.quality import LabelQualityAssessor
    qa = LabelQualityAssessor()
    report = qa.assess(label_path)
    click.echo(f"\n  Quality Grade: {report.get('quality_grade')} ({report.get('quality_score',0):.0%})")
    for check, data in report.get("checks", {}).items():
        icon = "✓" if data.get("status")=="ok" else "⚠"
        click.echo(f"  {icon} {check}: score={data.get('score','?')}")
    for rec in report.get("recommendations", []):
        click.echo(f"  ⟶ {rec}")
    if html:
        html_path = label_path.replace(".tif", "_quality.html")
        with open(html_path, "w") as f:
            f.write(qa.report_html(report))
        click.echo(f"  ✓ HTML report: {html_path}")
    click.echo()


# ── explain ───────────────────────────────────────────────────────────────────
@cli.group("explain")
def explain_grp():
    """Explainability — GradCAM, SHAP, uncertainty maps."""

@explain_grp.command("gradcam")
@click.argument("image_path")
@click.option("--model", "-m", required=True)
@click.option("--output", "-o", default=None)
@click.option("--class-idx", "-c", default=1, type=int)
def explain_gradcam(image_path, model, output, class_idx):
    """Generate GradCAM saliency map."""
    import pathlib
    output = output or image_path.replace(".tif", "_gradcam.tif")
    try:
        from pygeovision.models.registry import get_model
        from pygeovision.explainability.gradcam import GradCAM
        m = get_model(model)
        cam = GradCAM(m)
        result = cam.batch_explain(image_path, output, class_idx=class_idx)
        click.echo(f"  ✓ GradCAM → {output}")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)


# ── monitor ───────────────────────────────────────────────────────────────────
@cli.group("monitor")
def monitor_grp():
    """Model monitoring — drift detection, performance tracking."""

@monitor_grp.command("drift")
@click.argument("reference_dir")
@click.argument("current_dir")
@click.option("--output", "-o", default="./monitoring/drift_report.json")
def monitor_drift(reference_dir, current_dir, output):
    """Detect data distribution drift."""
    import glob, pathlib
    ref_images = glob.glob(f"{reference_dir}/*.tif")[:50]
    cur_images  = glob.glob(f"{current_dir}/*.tif")[:50]
    from pygeovision.monitoring.drift import DriftDetector
    d = DriftDetector()
    d.fit(ref_images)
    report = d.check(cur_images)
    pathlib.Path(output).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    level = report.get("data_drift", {}).get("drift_level", "?")
    click.echo(f"  Drift level: {level} | Report: {output}")


# ── pipeline ──────────────────────────────────────────────────────────────────
@cli.group("pipeline")
def pipeline_grp():
    """Pipeline management — create, run, schedule workflows."""

@pipeline_grp.command("run")
@click.argument("yaml_path")
@click.option("--dry-run", is_flag=True)
def pipeline_run(yaml_path, dry_run):
    """Run a YAML pipeline."""
    from pygeovision.pipelines.orchestrator import Pipeline
    click.echo(f"\n  Loading pipeline: {yaml_path}")
    try:
        p = Pipeline.from_yaml(yaml_path)
        click.echo(f"  Pipeline: {p.name} ({len(p._steps)} steps)")
        result = p.run(dry_run=dry_run)
        status = "✓" if result.success else "✗"
        click.echo(f"  {status} {result.steps_completed} completed in {result.duration_s}s\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)

@pipeline_grp.command("list-templates")
def pipeline_list_templates():
    """List available YAML pipeline templates."""
    import pathlib
    templates = list(pathlib.Path("/home/claude/pgv/pygeovision/pipelines/templates").glob("*.yaml"))
    click.echo(f"\n  Available templates ({len(templates)}):\n")
    for t in templates:
        click.echo(f"  • {t.stem:<25} ({t.name})")
    click.echo()


# ── edge ─────────────────────────────────────────────────────────────────────
@cli.group("edge")
def edge_grp():
    """Edge deployment — ONNX export, Jetson conversion."""

@edge_grp.command("export-onnx")
@click.argument("model_name")
@click.option("--output", "-o", default="./model.onnx")
@click.option("--classes", default=2, show_default=True, type=int)
@click.option("--in-channels", default=4, show_default=True, type=int)
@click.option("--input-size", default=512, show_default=True, type=int)
def edge_export_onnx(model_name, output, classes, in_channels, input_size):
    """Export a model to ONNX format."""
    click.echo(f"\n  Exporting {model_name} → ONNX ({output})")
    try:
        from pygeovision.models.registry import get_model
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        m = get_model(model_name, num_classes=classes, in_channels=in_channels)
        ONNXRuntimeInference.from_pytorch(
            m, output, input_shape=(1, in_channels, input_size, input_size), simplify=False
        )
        click.echo(f"  ✓ ONNX exported: {output}\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)

@edge_grp.command("benchmark-onnx")
@click.argument("onnx_path")
@click.option("--device", default="cpu", show_default=True)
@click.option("--runs", default=100, show_default=True, type=int)
def edge_benchmark(onnx_path, device, runs):
    """Benchmark ONNX model inference speed."""
    try:
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        eng = ONNXRuntimeInference(onnx_path, device=device)
        result = eng.benchmark(n_runs=runs)
        click.echo(f"\n  ONNX Benchmark ({onnx_path}):")
        click.echo(f"  Mean: {result['mean_ms']} ms | P95: {result['p95_ms']} ms | {result['fps']} FPS\n")
    except Exception as exc:
        click.echo(f"  ✗ {exc}", err=True)


# ── cloud ─────────────────────────────────────────────────────────────────────
@cli.group("cloud")
def cloud_grp():
    """Cloud deployment — AWS SageMaker, Azure ML, GCP Vertex AI."""

@cloud_grp.command("deploy-aws")
@click.argument("model_path")
@click.argument("endpoint_name")
@click.option("--region", default="us-east-1", show_default=True)
@click.option("--instance", default="ml.g4dn.xlarge", show_default=True)
def cloud_deploy_aws(model_path, endpoint_name, region, instance):
    """Deploy model to AWS SageMaker."""
    from pygeovision.cloud.deploy import AWSDeployer
    click.echo(f"\n  Deploying to AWS SageMaker: {endpoint_name}")
    result = AWSDeployer(region=region).deploy(model_path, endpoint_name, instance_type=instance)
    if result.get("success"):
        click.echo(f"  ✓ Endpoint live: {result.get('endpoint_url')}\n")
    else:
        click.echo(f"  ✗ {result.get('error')}\n", err=True)

@cloud_grp.command("deploy-gcp")
@click.argument("model_path")
@click.argument("endpoint_name")
@click.option("--project", required=True)
@click.option("--region", default="us-central1")
def cloud_deploy_gcp(model_path, endpoint_name, project, region):
    """Deploy model to GCP Vertex AI."""
    from pygeovision.cloud.deploy import GCPDeployer
    click.echo(f"\n  Deploying to GCP Vertex AI: {endpoint_name}")
    result = GCPDeployer(project_id=project, region=region).deploy(model_path, endpoint_name)
    click.echo(f"  {'✓' if result.get('success') else '✗'} {result}\n")


# ── vlm ───────────────────────────────────────────────────────────────────────
@cli.group("vlm")
def vlm_grp():
    """Vision-Language Models — caption, query, search imagery."""

@vlm_grp.command("caption")
@click.argument("image_path")
def vlm_caption(image_path):
    """Generate a natural language caption for a satellite image."""
    from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
    click.echo(f"\n  Generating caption for: {image_path}")
    caption = MoondreamGeo().caption(image_path)
    click.echo(f"  Caption: {caption}\n")

@vlm_grp.command("query")
@click.argument("image_path")
@click.argument("question")
def vlm_query(image_path, question):
    """Answer a question about a satellite image."""
    from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
    answer = MoondreamGeo().vqa(image_path, question)
    click.echo(f"  A: {answer}\n")

@vlm_grp.command("search")
@click.argument("query")
@click.argument("image_dir")
@click.option("--top-k", default=5, show_default=True, type=int)
def vlm_search(query, image_dir, top_k):
    """Search a directory of images by text query."""
    from pygeovision.advanced.vlm.clip_geo import CLIPGeo
    results = CLIPGeo().search(query, image_dir, top_k=top_k)
    click.echo(f"\n  Top {len(results)} results for '{query}':")
    for r in results:
        click.echo(f"  {r['score']:.3f}  {r['path']}")
    click.echo()


# ── timeseries ────────────────────────────────────────────────────────────────
@cli.group("timeseries")
def timeseries_grp():
    """Time series analysis — NDVI trends, anomaly detection."""

@timeseries_grp.command("analyze")
@click.argument("images", nargs=-1, required=True)
@click.option("--index", "-i", default="ndvi", show_default=True,
               type=click.Choice(["ndvi","ndwi","evi","savi","ndbi"]))
@click.option("--sensor", "-s", default="sentinel2", show_default=True)
@click.option("--output", "-o", default=None)
def ts_analyze(images, index, sensor, output):
    """Compute spectral index time series."""
    from pygeovision.advanced.timeseries import GeoTimeSeries
    ts = GeoTimeSeries(sensor=sensor)
    series = ts.compute_index_series(list(images), index=index)
    click.echo(f"\n  {index.upper()} series: {len(images)} images")
    vals = series.get("mean", [])
    for i, v in enumerate(vals):
        click.echo(f"  [{i+1:>3}] {v:.4f}" if v is not None else f"  [{i+1:>3}] N/A")
    trend = ts.compute_trend(series)
    click.echo(f"  Trend: {trend.get('direction','?')} (slope={trend.get('slope',0):.6f})\n")
    if output:
        import json
        with open(output, "w") as f:
            json.dump(series, f, indent=2)
        click.echo(f"  ✓ Saved: {output}\n")

@timeseries_grp.command("anomaly")
@click.argument("images", nargs=-1, required=True)
@click.option("--threshold", "-t", default=2.5, show_default=True, type=float)
def ts_anomaly(images, threshold):
    """Detect anomalous dates in a time series."""
    from pygeovision.advanced.timeseries import GeoTimeSeries
    ts = GeoTimeSeries()
    series = ts.compute_index_series(list(images), "ndvi")
    anomalies = ts.detect_anomalies(series, threshold=threshold)
    if anomalies:
        click.echo(f"\n  Found {len(anomalies)} anomalies (z>{threshold}):")
        for a in anomalies:
            click.echo(f"  {a['date']}: value={a['value']:.4f} z={a['zscore']:.2f} ({a['type']})")
    else:
        click.echo("  No anomalies detected.")
    click.echo()

def main() -> None:
    """CLI entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()


# ═══════════════════════════════════════════════════════════════════════
# DATASETS commands (Phase 6.2)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def datasets() -> None:
    """Browse, search, and download 500+ remote sensing datasets (EarthNets)."""


@datasets.command("list")
@click.option("--domain", default=None, help="Filter by research domain.")
@click.option("--modality", default=None, help="Filter by data modality.")
@click.option("--task", default=None, help="Filter by ML task (segmentation, detection, …).")
@click.option("--min-year", default=None, type=int)
@click.option("--max-volume-gb", default=None, type=float, help="Max dataset size in GB.")
@click.option("--max-res-m", default=None, type=float, help="Max spatial resolution in metres.")
@click.option("--limit", default=25, show_default=True)
def datasets_list(domain, modality, task, min_year, max_volume_gb, max_res_m, limit):
    """List datasets from the 500+ EarthNets catalog."""
    from pygeovision.datasets.registry import dataset_registry
    items = dataset_registry.filter(
        domain=domain, modality=modality, task=task,
        min_year=min_year, max_volume_gb=max_volume_gb,
        max_resolution_m=max_res_m,
    )
    items = items[:limit]
    dataset_registry.print_table(items)


@datasets.command("search")
@click.argument("query")
@click.option("--limit", default=20, show_default=True)
def datasets_search(query, limit):
    """Search datasets by keyword (name, description, domain, task)."""
    from pygeovision.datasets.registry import dataset_registry
    results = dataset_registry.search(query)[:limit]
    if not results:
        click.echo(f"No datasets found for '{query}'.")
        return
    click.echo(f"\nFound {len(results)} dataset(s) matching '{query}':\n")
    dataset_registry.print_table(results)


@datasets.command("info")
@click.argument("name")
def datasets_info(name):
    """Show full metadata for a dataset."""
    from pygeovision.datasets.loader import DatasetLoader
    DatasetLoader().info(name)


@datasets.command("download")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output directory.")
def datasets_download(name, output):
    """Download and extract a dataset by name."""
    from pygeovision.datasets.loader import DatasetLoader
    DatasetLoader().download(name, output)


@datasets.command("top")
@click.argument("task", default="segmentation")
@click.option("--n", default=5, show_default=True, help="Number of top datasets to show.")
def datasets_top(task, n):
    """Show the top-N datasets for a given ML task (EarthNets ranking).

    \b
    Tasks: segmentation, detection, classification, change_detection,
           multi_label, regression, prediction, self_supervised, vqa
    """
    from pygeovision.datasets.registry import dataset_registry
    top = dataset_registry.top_for_task(task, n=n)
    if not top:
        click.echo(f"No datasets found for task '{task}'.")
        return
    click.echo(f"\nTop-{n} datasets for '{task}' (EarthNets ranking):\n")
    dataset_registry.print_table(top)


@datasets.command("similar")
@click.argument("name")
@click.option("--n", default=10, show_default=True)
def datasets_similar(name, n):
    """Find datasets similar to NAME using the EarthNets similarity formula."""
    from pygeovision.datasets.registry import dataset_registry
    similar = dataset_registry.similar_to(name, n=n)
    click.echo(f"\nDatasets most similar to '{name}':\n")
    dataset_registry.print_table(similar)


@datasets.command("domains")
def datasets_domains():
    """List all research domains in the catalog."""
    from pygeovision.datasets.registry import dataset_registry
    s = dataset_registry.summary()
    click.echo(f"\nDatasets by domain ({len(s['domains'])} domains):\n")
    for domain, count in sorted(s["domains"].items(), key=lambda x: -x[1]):
        click.echo(f"  {domain:<25} {count:>4} datasets")


@datasets.command("stats")
def datasets_stats():
    """Show catalog statistics (total datasets, volume, domains, tasks)."""
    from pygeovision.datasets.registry import dataset_registry
    s = dataset_registry.summary()
    click.echo(f"\n{'═'*50}")
    click.echo(f"  PyGeoVision Dataset Catalog (EarthNets)")
    click.echo(f"{'═'*50}")
    click.echo(f"  Total datasets : {s['total_datasets']}")
    click.echo(f"  Total volume   : {s['total_volume_tb']} TB")
    click.echo(f"  Year range     : {s['year_range'][0]} – {s['year_range'][1]}")
    click.echo(f"  Domains        : {len(s['domains'])}")
    click.echo(f"  Unique tasks   : {len(s['tasks'])}")
    click.echo(f"\n  Tasks: {', '.join(sorted(s['tasks']))}")


# ═══════════════════════════════════════════════════════════════════════
# MODELS zoo commands (Phase 2 / Phase 6.3)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def zoo() -> None:
    """Browse the AI Model Zoo (98+ architectures across all geospatial tasks)."""


@zoo.command("list")
@click.option("--task", default=None, help="Filter by task (segmentation, detection, …).")
@click.option("--tag", default=None, help="Filter by tag (transformer, real_time, sar, …).")
@click.option("--max-params", default=None, type=float, help="Max parameters in millions.")
@click.option("--hf-only", is_flag=True, help="Only models with HuggingFace weights.")
def zoo_list(task, tag, max_params, hf_only):
    """List models from the AI Model Zoo."""
    from pygeovision.ai.models.zoo import model_zoo
    items = model_zoo.filter(task=task, tag=tag, max_params_m=max_params)
    if hf_only:
        items = [m for m in items if m.hf_model_id]
    model_zoo.print_table(items)


@zoo.command("search")
@click.argument("query")
def zoo_search(query):
    """Search the model zoo by name, architecture, or tag."""
    from pygeovision.ai.models.zoo import model_zoo
    results = model_zoo.search(query)
    if not results:
        click.echo(f"No models found for '{query}'.")
        return
    model_zoo.print_table(results)


@zoo.command("top")
@click.argument("task")
@click.option("--n", default=5, show_default=True)
def zoo_top(task, n):
    """Show the top-N models for a task.

    \b
    Tasks: segmentation, detection, classification, change_detection,
           foundation, vlm, 3d, timeseries, super_resolution
    """
    from pygeovision.ai.models.zoo import model_zoo
    top = model_zoo.top_for_task(task, n=n)
    click.echo(f"\nTop-{n} models for '{task}':\n")
    model_zoo.print_table(top)


@zoo.command("info")
@click.argument("name")
def zoo_info(name):
    """Show full metadata for a model."""
    from pygeovision.ai.models.zoo import model_zoo
    try:
        m = model_zoo[name]
        click.echo(f"\nModel: {m.name}")
        click.echo(f"  Task:          {m.task}")
        click.echo(f"  Architecture:  {m.architecture}")
        click.echo(f"  Backbone:      {m.backbone}")
        click.echo(f"  Parameters:    {m.params_m:.1f}M")
        click.echo(f"  Input bands:   {m.input_bands}")
        click.echo(f"  Pretrained:    {'Yes' if m.pretrained_available else 'No'}")
        if m.hf_model_id:
            click.echo(f"  HF Hub ID:     {m.hf_model_id}")
        if m.description:
            click.echo(f"  Description:   {m.description}")
        if m.tags:
            click.echo(f"  Tags:          {', '.join(m.tags)}")
        if m.paper_url:
            click.echo(f"  Paper:         {m.paper_url}")
    except KeyError as exc:
        click.echo(f"Error: {exc}", err=True)


@zoo.command("stats")
def zoo_stats():
    """Show model zoo statistics."""
    from pygeovision.ai.models.zoo import model_zoo
    s = model_zoo.summary()
    click.echo(f"\n{'═'*50}")
    click.echo(f"  PyGeoVision AI Model Zoo")
    click.echo(f"{'═'*50}")
    click.echo(f"  Total models      : {s['total_models']}")
    click.echo(f"  With HF weights   : {s['with_hf_weights']}")
    click.echo(f"  Pretrained avail  : {s['pretrained']}")
    click.echo(f"\n  By task:")
    for task, count in sorted(s["tasks"].items(), key=lambda x: -x[1]):
        click.echo(f"    {task:<22} {count:>3} models")


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARK commands (Phase 4.4 / Phase 6.5)
# ═══════════════════════════════════════════════════════════════════════

@cli.group()
def benchmark() -> None:
    """Benchmarking tools — evaluate models, compare results, generate leaderboards."""


@benchmark.command("datasets")
@click.argument("task")
@click.option("--n", default=5, show_default=True, help="Number of top datasets to show.")
def benchmark_datasets(task, n):
    """Select the best benchmark datasets for a given task (EarthNets top-5).

    \b
    Example:
        pygeovision benchmark datasets segmentation --n 5
        pygeovision benchmark datasets detection
        pygeovision benchmark datasets change_detection
    """
    from pygeovision.datasets.registry import dataset_registry
    top = dataset_registry.top_for_task(task, n=n)
    if not top:
        click.echo(f"No datasets found for task '{task}'.")
        return
    click.echo(f"\n{'═'*70}")
    click.echo(f"  Recommended Benchmark Datasets for '{task}' (EarthNets Top-{n})")
    click.echo(f"{'═'*70}")
    dataset_registry.print_table(top)


@benchmark.command("models")
@click.argument("task")
@click.option("--n", default=5, show_default=True)
def benchmark_models(task, n):
    """Show the top-N recommended models to benchmark for a task."""
    from pygeovision.ai.models.zoo import model_zoo
    top = model_zoo.top_for_task(task, n=n)
    if not top:
        click.echo(f"No models found for task '{task}'.")
        return
    click.echo(f"\nRecommended models for '{task}' benchmarks:")
    model_zoo.print_table(top)


@benchmark.command("tasks")
def benchmark_tasks():
    """List all available benchmark tasks."""
    from pygeovision.datasets.registry import dataset_registry
    from pygeovision.ai.models.zoo import model_zoo
    d_tasks = set(dataset_registry.tasks())
    m_tasks = set(model_zoo.tasks())
    all_tasks = sorted(d_tasks | m_tasks)
    click.echo(f"\nAvailable benchmark tasks ({len(all_tasks)}):\n")
    for t in all_tasks:
        d_count = len(dataset_registry.filter(task=t))
        m_count = len(model_zoo.filter(task=t))
        click.echo(f"  {t:<25} {d_count:>4} datasets   {m_count:>4} models")


# ═══════════════════════════════════════════════════════════════════════
# VALIDATE command group — mandatory data validation
# ═══════════════════════════════════════════════════════════════════════

@cli.group("validate")
def validate_grp():
    """Validate satellite data before model inference."""


@validate_grp.command("run")
@click.argument("input_path")
@click.option("--required-bands", type=int, default=None, help="Minimum number of bands required.")
@click.option("--value-range", default=None, help="Expected value range, e.g. '0,10000'.")
@click.option("--target-crs", default=None, help="Expected CRS, e.g. EPSG:4326.")
@click.option("--report", default=None, help="Save HTML/JSON report to this path.")
@click.option("--report-fmt", default="html", type=click.Choice(["html","json"]), show_default=True)
@click.option("--mode", default="fix", type=click.Choice(["strict","fix","warn"]), show_default=True)
def validate_run(input_path, required_bands, value_range, target_crs, report, report_fmt, mode):
    """Validate a GeoTIFF — check nulls, dtype, CRS, bounds, outliers.

    \b
    Examples:
        pygeovision validate run scene.tif
        pygeovision validate run scene.tif --required-bands 6 --value-range 0,10000
        pygeovision validate run scene.tif --report validation.html
    """
    from pygeovision.data.validator import DataValidator
    v = DataValidator(mode=mode)
    vr = None
    if value_range:
        lo, hi = [float(x) for x in value_range.split(",")]
        vr = (lo, hi)
    result = v.validate(input_path, required_bands=required_bands,
                        value_range=vr, target_crs=target_crs)
    click.echo(result.summary())
    if report:
        v.generate_report(input_path, report, fmt=report_fmt)
        click.echo(f"\n  Report → {report}")


@validate_grp.command("for-inference")
@click.argument("input_path")
@click.option("--model-type", default="segmentation",
              type=click.Choice(["segmentation","detection","change_detection",
                                 "classification","foundation","regression"]),
              show_default=True)
def validate_for_inference(input_path, model_type):
    """Validate and auto-fix data for a specific model type.

    \b
    Example:
        pygeovision validate for-inference scene.tif --model-type segmentation
    """
    from pygeovision.data.validator import DataValidator
    v = DataValidator(mode="fix")
    arr = v.validate_for_inference(input_path, model_type=model_type)
    click.echo(f"  ✓ Ready for {model_type}  |  shape={arr.shape}  "
               f"dtype={arr.dtype}  range=[{arr.min():.4f}, {arr.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════════
# PREPROCESS command group
# ═══════════════════════════════════════════════════════════════════════

@cli.group("preprocess")
def preprocess_grp():
    """Satellite image preprocessing — stack, clip, mask, normalise, resample."""


@preprocess_grp.command("stack")
@click.argument("scene_dir")
@click.option("--bands", required=True, help="Comma-separated band names, e.g. B02,B03,B04,B08")
@click.option("--output", required=True, help="Output stacked GeoTIFF path.")
@click.option("--validate", is_flag=True, default=False, help="Validate output after stacking.")
def preprocess_stack(scene_dir, bands, output, validate):
    """Stack individual band files from a scene directory.

    \b
    Example:
        pygeovision preprocess stack ./S2C_20240628/ --bands B02,B03,B04,B08,B11,B12 --output stack.tif
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    band_list = [b.strip() for b in bands.split(",")]
    out = pre.stack_from_dir(scene_dir, band_list, output)
    click.echo(f"  ✓ Stacked {len(band_list)} bands → {out}")
    if validate:
        report = pre.validate(out)
        click.echo(report.summary())


@preprocess_grp.command("clip")
@click.argument("input_path")
@click.option("--bbox", required=True, help="minlon,minlat,maxlon,maxlat in WGS84.")
@click.option("--output", default=None, help="Output path (default: _clipped).")
@click.option("--validate", is_flag=True, default=False)
def preprocess_clip(input_path, bbox, output, validate):
    """Clip a raster to a bounding box.

    \b
    Example:
        pygeovision preprocess clip stack.tif --bbox -74.1,40.6,-73.7,40.9
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    coords = tuple(float(x) for x in bbox.split(","))
    out = pre.clip_to_bbox(input_path, coords, output_path=output)
    click.echo(f"  ✓ Clipped → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("normalise")
@click.argument("input_path")
@click.option("--method", default="minmax",
              type=click.Choice(["minmax","zscore","percentile","scale_factor"]),
              show_default=True)
@click.option("--scale-factor", default=10000.0, show_default=True,
              help="Divisor for scale_factor method (Sentinel-2 L2A = 10000).")
@click.option("--percentile", default=2.0, show_default=True)
@click.option("--output", default=None)
@click.option("--validate", is_flag=True, default=False)
def preprocess_normalise(input_path, method, scale_factor, percentile, output, validate):
    """Normalise pixel values.

    \b
    Example:
        pygeovision preprocess normalise stack.tif --method scale_factor --scale-factor 10000
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    out = pre.normalise(input_path, method=method, output_path=output,
                         scale_factor=scale_factor, percentile=percentile)
    click.echo(f"  ✓ Normalised ({method}) → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("resample")
@click.argument("input_path")
@click.option("--resolution", required=True, type=float, help="Target pixel size in metres.")
@click.option("--output", default=None)
@click.option("--method", default="bilinear",
              type=click.Choice(["nearest","bilinear","cubic","lanczos","average"]),
              show_default=True)
@click.option("--validate", is_flag=True, default=False)
def preprocess_resample(input_path, resolution, output, method, validate):
    """Resample a raster to a target resolution.

    \b
    Example:
        pygeovision preprocess resample s2_20m.tif --resolution 10
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    out = pre.resample(input_path, resolution, output_path=output, resampling=method)
    click.echo(f"  ✓ Resampled to {resolution}m → {out}")
    if validate:
        click.echo(pre.validate(out).summary())


@preprocess_grp.command("pipeline")
@click.argument("input_path")
@click.option("--output", required=True, help="Final output path.")
@click.option("--bands", default=None, help="Bands to stack, e.g. B02,B03,B04,B08,B11,B12")
@click.option("--bbox", default=None, help="Clip bbox: minlon,minlat,maxlon,maxlat")
@click.option("--scl", default=None, help="SCL band path for cloud masking.")
@click.option("--normalise", default=None,
              type=click.Choice(["minmax","zscore","percentile","scale_factor","none"]))
@click.option("--resample-m", default=None, type=float, help="Resample to N metres.")
@click.option("--validate", is_flag=True, default=False)
def preprocess_pipeline(input_path, output, bands, bbox, scl, normalise, resample_m, validate):
    """Run the full preprocessing pipeline in one command.

    \b
    Example:
        pygeovision preprocess pipeline ./S2C_20240628/ \\
            --output ready.tif \\
            --bands B02,B03,B04,B08,B11,B12 \\
            --bbox -74.1,40.6,-73.7,40.9 \\
            --scl SCL.tif \\
            --normalise scale_factor
    """
    from pygeovision.preprocess import Preprocessor
    pre = Preprocessor()
    band_list = [b.strip() for b in bands.split(",")] if bands else None
    bbox_t    = tuple(float(x) for x in bbox.split(",")) if bbox else None
    norm_val  = None if (not normalise or normalise == "none") else normalise

    result = pre.pipeline(
        input_path=input_path,
        output_path=output,
        stack_bands=band_list,
        bbox=bbox_t,
        scl_path=scl,
        normalise=norm_val,
        resample_m=resample_m,
    )
    click.echo(f"  ✓ Pipeline complete")
    click.echo(f"    Output  : {result['output_path']}")
    click.echo(f"    Shape   : {result['shape']}")
    click.echo(f"    Res     : {result['resolution_m']:.1f}m")
    click.echo(f"    Steps   : {' → '.join(result['steps_applied'])}")
    if validate:
        click.echo(pre.validate(result["output_path"]).summary())


# ═══════════════════════════════════════════════════════════════════════
# INDICES command group — 22 spectral indices
# ═══════════════════════════════════════════════════════════════════════

@cli.group("indices")
def indices_grp():
    """Compute validated spectral indices (NDVI, EVI, NBR, TCT, PCA …)."""


@indices_grp.command("compute")
@click.argument("input_path")
@click.argument("index_name")
@click.option("--output", default=None, help="Output GeoTIFF path.")
@click.option("--red", default=3, show_default=True, type=int, help="1-based red band index.")
@click.option("--nir", default=4, show_default=True, type=int, help="1-based NIR band index.")
@click.option("--green", default=2, show_default=True, type=int)
@click.option("--blue", default=1, show_default=True, type=int)
@click.option("--swir1", default=5, show_default=True, type=int)
@click.option("--swir2", default=6, show_default=True, type=int)
@click.option("--validate", is_flag=True, default=False)
def indices_compute(input_path, index_name, output, red, nir, green, blue, swir1, swir2, validate):
    """Compute a spectral index.

    INDEX_NAME: ndvi, evi, savi, msavi, ndwi, mndwi, ndbi, bsi, nbr, bai, ndsi, rvi, arvi, lswi, ndre, wdrvi, vari, exg

    \b
    Examples:
        pygeovision indices compute stack.tif ndvi --output ndvi.tif
        pygeovision indices compute stack.tif nbr  --nir 4 --swir2 6
    """
    from pygeovision.data.indices import SpectralIndices
    ix = SpectralIndices()
    bm = dict(blue=blue, green=green, red=red, nir=nir, swir1=swir1, swir2=swir2)

    fn_map = {
        "ndvi": lambda: ix.ndvi(input_path, red, nir, output),
        "evi":  lambda: ix.evi( input_path, blue, red, nir, output_path=output),
        "savi": lambda: ix.savi(input_path, red, nir, output),
        "msavi":lambda: ix.msavi(input_path, red, nir, output),
        "ndwi": lambda: ix.ndwi(input_path, green, nir, output),
        "mndwi":lambda: ix.mndwi(input_path, green, swir1, output),
        "ndbi": lambda: ix.ndbi(input_path, swir1, nir, output),
        "bsi":  lambda: ix.bsi( input_path, blue, red, nir, output_path=output),
        "nbr":  lambda: ix.nbr( input_path, nir, swir2, output),
        "bai":  lambda: ix.bai( input_path, red, nir, output),
        "ndsi": lambda: ix.ndsi(input_path, green, swir1, output),
        "rvi":  lambda: ix.rvi( input_path, red, nir, output),
        "arvi": lambda: ix.arvi(input_path, blue, red, nir, output_path=output),
        "lswi": lambda: ix.lswi(input_path, nir, swir1, output),
        "ndre": lambda: ix.ndre(input_path, swir1, nir, output),
        "wdrvi":lambda: ix.wdrvi(input_path, red, nir, output),
        "vari": lambda: ix.vari(input_path, blue, green, red, output),
        "exg":  lambda: ix.exg( input_path, blue, green, red, output),
    }
    idx = index_name.lower()
    if idx not in fn_map:
        click.echo(f"Error: unknown index '{index_name}'. Available: {sorted(fn_map)}", err=True)
        return

    result = fn_map[idx]()
    if isinstance(result, str):
        click.echo(f"  ✓ {index_name.upper()} → {result}")
    else:
        import numpy as np
        click.echo(f"  ✓ {index_name.upper()}  shape={result.shape}  "
                   f"range=[{float(result.min()):.4f}, {float(result.max()):.4f}]")


@indices_grp.command("all")
@click.argument("input_path")
@click.option("--output-dir", required=True, help="Directory for all index GeoTIFFs.")
@click.option("--indices", default=None,
              help="Comma-separated list. Default: all 16 standard indices.")
def indices_all(input_path, output_dir, indices):
    """Compute all (or selected) spectral indices in one command.

    \b
    Example:
        pygeovision indices all stack.tif --output-dir ./indices/
        pygeovision indices all stack.tif --output-dir ./indices/ --indices ndvi,evi,nbr,ndwi
    """
    from pygeovision.data.indices import SpectralIndices
    ix = SpectralIndices()
    idx_list = [i.strip() for i in indices.split(",")] if indices else None
    results  = ix.compute_all(input_path, indices=idx_list, output_dir=output_dir)
    click.echo(f"\n  Computed {len(results)} indices → {output_dir}")
    for name, path in results.items():
        click.echo(f"    {name:<8} → {path}")


@indices_grp.command("list")
def indices_list():
    """List all available spectral indices."""
    indices = [
        ("ndvi",  "Normalized Difference Vegetation Index",  "Vegetation"),
        ("evi",   "Enhanced Vegetation Index",               "Vegetation"),
        ("savi",  "Soil-Adjusted Vegetation Index",          "Vegetation"),
        ("msavi", "Modified SAVI",                           "Vegetation"),
        ("arvi",  "Atmospherically Resistant Vegetation Index","Vegetation"),
        ("ndre",  "Normalized Difference Red Edge",          "Vegetation"),
        ("rvi",   "Ratio Vegetation Index",                  "Vegetation"),
        ("wdrvi", "Wide Dynamic Range Vegetation Index",     "Vegetation"),
        ("vari",  "Visible Atmospherically Resistant Index", "Vegetation/RGB"),
        ("exg",   "Excess Green Index",                      "Vegetation/RGB"),
        ("ndwi",  "Normalized Difference Water Index",       "Water"),
        ("mndwi", "Modified NDWI",                           "Water"),
        ("lswi",  "Land Surface Water Index",                "Water"),
        ("wri",   "Water Ratio Index",                       "Water"),
        ("ndbi",  "Normalized Difference Built-up Index",    "Urban"),
        ("bsi",   "Bare Soil Index",                         "Soil"),
        ("nbr",   "Normalized Burn Ratio",                   "Fire"),
        ("bai",   "Burn Area Index",                         "Fire"),
        ("ndsi",  "Normalized Difference Snow Index",        "Snow"),
        ("tct",   "Tasseled Cap Transform (B/G/W)",          "Transform"),
        ("pca",   "Principal Component Analysis",            "Transform"),
    ]
    click.echo(f"\n  {'Index':<8} {'Name':<42} Category")
    click.echo(f"  {'─'*8} {'─'*42} {'─'*16}")
    for name, desc, cat in indices:
        click.echo(f"  {name:<8} {desc:<42} {cat}")
    click.echo(f"\n  Total: {len(indices)} indices")


# ═══════════════════════════════════════════════════════════════════════
# POSTPROCESS command group
# ═══════════════════════════════════════════════════════════════════════

@cli.group("postprocess")
def postprocess_grp():
    """Postprocess model predictions — vectorise, smooth, statistics, export."""


@postprocess_grp.command("vectorise")
@click.argument("input_path")
@click.option("--output", required=True, help="Output GeoJSON path.")
@click.option("--target-class", default=None, type=int, help="Only vectorise this class.")
@click.option("--min-area", default=0.0, type=float, help="Minimum polygon area in m².")
@click.option("--simplify", default=0.0, type=float, help="Douglas-Peucker tolerance.")
@click.option("--validate", is_flag=True, default=False)
def post_vectorise(input_path, output, target_class, min_area, simplify, validate):
    """Vectorise a classification raster to GeoJSON polygons.

    \b
    Example:
        pygeovision postprocess vectorise pred.tif --output buildings.geojson --target-class 1 --min-area 50
    """
    from pygeovision.data.postprocess import PostProcessor
    post = PostProcessor()
    out = post.vectorise(input_path, output, target_class=target_class,
                          min_area_m2=min_area, simplify_tolerance=simplify)
    import json
    with open(out) as f:
        n = len(json.load(f).get("features", []))
    click.echo(f"  ✓ Vectorised → {out}  ({n} features)")


@postprocess_grp.command("sieve")
@click.argument("input_path")
@click.option("--min-pixels", default=10, show_default=True, type=int)
@click.option("--output", default=None)
def post_sieve(input_path, min_pixels, output):
    """Remove small patches from a classification raster.

    \b
    Example:
        pygeovision postprocess sieve pred.tif --min-pixels 10
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().sieve_filter(input_path, min_pixels, output_path=output)
    click.echo(f"  ✓ Sieve (min={min_pixels}px) → {out}")


@postprocess_grp.command("smooth")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--tolerance", default=0.5, show_default=True, type=float)
def post_smooth(input_path, output, tolerance):
    """Smooth vector polygon boundaries.

    \b
    Example:
        pygeovision postprocess smooth buildings.geojson --output smooth.geojson --tolerance 0.5
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().smooth(input_path, output, tolerance=tolerance)
    click.echo(f"  ✓ Smoothed → {out}")


@postprocess_grp.command("regularise")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--min-area", default=10.0, type=float, show_default=True)
def post_regularise(input_path, output, min_area):
    """Regularise building footprints to minimum rotated rectangles.

    \b
    Example:
        pygeovision postprocess regularise buildings.geojson --output regular.geojson
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().regularise_buildings(input_path, output, min_area_m2=min_area)
    click.echo(f"  ✓ Regularised → {out}")


@postprocess_grp.command("zonal-stats")
@click.argument("raster_path")
@click.argument("vector_path")
@click.option("--output", required=True)
@click.option("--stats", default="mean,std,min,max,count", show_default=True)
def post_zonal_stats(raster_path, vector_path, output, stats):
    """Compute per-zone raster statistics.

    \b
    Example:
        pygeovision postprocess zonal-stats ndvi.tif fields.geojson --output stats.geojson
    """
    from pygeovision.data.postprocess import PostProcessor
    stat_list = [s.strip() for s in stats.split(",")]
    out = PostProcessor().zonal_statistics(raster_path, vector_path, output, stats=stat_list)
    click.echo(f"  ✓ Zonal statistics → {out}")


@postprocess_grp.command("accuracy")
@click.argument("prediction_path")
@click.argument("reference_path")
@click.option("--class-names", default=None, help="Comma-separated class names.")
def post_accuracy(prediction_path, reference_path, class_names):
    """Compute accuracy metrics vs a reference raster.

    \b
    Example:
        pygeovision postprocess accuracy pred.tif ref.tif --class-names Background,Building,Road
    """
    from pygeovision.data.postprocess import PostProcessor
    names = [n.strip() for n in class_names.split(",")] if class_names else None
    report = PostProcessor().accuracy_assessment(prediction_path, reference_path, class_names=names)
    click.echo(f"\n  Overall Accuracy : {report['overall_accuracy']:.4f}")
    click.echo(f"  Cohen's κ        : {report['kappa']:.4f}")
    click.echo(f"  Mean IoU         : {report['mean_iou']:.4f}")
    click.echo(f"  Total pixels     : {report['total_pixels']:,}")
    click.echo(f"\n  Per-class metrics:")
    click.echo(f"  {'Class':<6} {'Name':<16} {'Precision':>10} {'Recall':>8} {'F1':>8} {'IoU':>8}")
    click.echo(f"  {'─'*6} {'─'*16} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    names_list = report.get("class_names", [])
    for i, (cls, m) in enumerate(report["per_class"].items()):
        nm = names_list[i] if i < len(names_list) else str(cls)
        click.echo(f"  {cls:<6} {nm:<16} {m['precision']:>10.4f} "
                   f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['iou']:>8.4f}")


@postprocess_grp.command("class-stats")
@click.argument("prediction_path")
def post_class_stats(prediction_path):
    """Print per-class pixel count and area statistics.

    \b
    Example:
        pygeovision postprocess class-stats prediction.tif
    """
    from pygeovision.data.postprocess import PostProcessor
    stats = PostProcessor().class_statistics(prediction_path)
    click.echo(f"\n  {'Class':>6} {'Pixels':>12} {'Area (ha)':>12} {'Area km²':>12} {'Pct':>8}")
    click.echo(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*12} {'─'*8}")
    for cls, info in sorted(stats.items()):
        click.echo(f"  {cls:>6} {info['pixels']:>12,} {info['area_ha']:>12.3f} "
                   f"{info['area_km2']:>12.5f} {info['pct']:>7.2f}%")


@postprocess_grp.command("cog")
@click.argument("input_path")
@click.option("--output", default=None)
def post_cog(input_path, output):
    """Convert a GeoTIFF to Cloud-Optimized GeoTIFF.

    \b
    Example:
        pygeovision postprocess cog prediction.tif --output prediction_cog.tif
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().to_cog(input_path, output_path=output)
    click.echo(f"  ✓ COG → {out}")


@postprocess_grp.command("export")
@click.argument("input_path")
@click.option("--output", required=True)
@click.option("--format", "fmt", default="geojson",
              type=click.Choice(["geojson","shp","gpkg","kml"]), show_default=True)
def post_export(input_path, output, fmt):
    """Export vector to GeoJSON, Shapefile, GeoPackage, or KML.

    \b
    Example:
        pygeovision postprocess export buildings.geojson --output buildings.gpkg --format gpkg
    """
    from pygeovision.data.postprocess import PostProcessor
    out = PostProcessor().export(input_path, output, fmt=fmt)
    click.echo(f"  ✓ Exported ({fmt.upper()}) → {out}")
