# Recovered code-only snapshots

`kaggle-v2-cpu/` preserves the 26 source/configuration/provenance files from a local Kaggle CPU code package. The original ZIP remains private. Its SHA-256 is `17341a1112f8a86a59acad9c78146851a47ed55024dfa3410fd1e7f11bae3d7e`.

The snapshot includes Python source, a notebook with no saved outputs, dependency files, registry metadata, protocol/design documents and the original per-file package manifest. It excludes manuscript PDFs/ZIPs, raw/processed datasets, frozen results, credentials and correspondence.

Most source versions already exist on the repository's current branches, but the package's exact `src/v2_engineering.py` version was not reachable from the preserved `main`/V4 histories. This snapshot keeps that otherwise-local code available without replacing current modules or claiming it is the preferred implementation.

Every extracted source file is preserved byte-for-byte and checked against the original package manifest. All Python files were compiled in memory for syntax verification. No Kaggle execution, model training or runtime equivalence is claimed by this preservation task. The historical package README and protocol documents describe their original workflow, not current publication approval.
