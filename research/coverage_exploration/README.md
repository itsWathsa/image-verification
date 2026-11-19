# Archived COVERAGE exploration helpers

These scripts preserve the interrupted research work. None is imported by the app, run by CI, or a validated edit detector. See ../../HANDOFF.md for the current status and next steps.

They inspect public download sources, dataset-viewer metadata and archive layouts. Outputs go into the ignored `.models/coverage/` cache. The native archive is approximately 5.6 GB and contains unrelated benchmarks; its transfer failed locally. The compact Parquet mirror was downloaded but its native readers were blocked by Windows Application Control. Viewer assets were saved for exploration and may be JPEG-reencoded.

Do not run the entire directory as a pipeline. Some scripts were unsuccessful probes, sparse queries reached rate limits, and `locate_coverage_parquet.py` requires a working Parquet reader. Preserve these failures as context rather than assuming the scripts completed an evaluation. No credentials, dataset image bytes or private user photos are included here.
