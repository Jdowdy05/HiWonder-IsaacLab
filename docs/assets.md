# Assets and licensing

[Documentation](README.md) · [Project overview](../README.md)

Generated robot USDs, scale variants, registration/camera manifests, collision profiles and the stock stable-grasp seed are included under the extension's `isaaclab_hiwonder/assets/`. Stock assets live in `generated/`; the explicit Eka model lives in `eka_v6/`. The original large stock CAD conversion input is identified by hash in its manifest and is not bundled. The Eka directory includes the supplied replacement `source.step`, SHA-256 `141ab8a721d850ed88da9d97f17257ebce1f85cfca7a0403f4c6e20c1f60e944`, preserved unchanged.

The Eka derivative uses exact NURBS reparameterization for 13 printed parts and boundary patches for two caps. Those caps remain faceted CAD; their original parametric construction was not recovered. Geometry checks found the resulting printable meshes watertight, consistently wound and positive-volume at actual USD float precision across nominal/0.99/1.01 scale, with retained contact triangles. Earlier native V6 data supplies reference hardware/case meshes only; changed printed geometry comes from the new STEP. Matching four-hole mounting patterns on a 14 mm pitch circle place the hand after `wrist_roll`; physical fastener fit, joint travel and dynamics remain unmeasured. These geometry checks do not establish physical assembly or contact acceptance.

Software retains its existing BSD-3-Clause headers; see [LICENSE](../LICENSE). Source CAD and derived asset redistribution rights have not been established. See [ASSET_PROVENANCE.md](../ASSET_PROVENANCE.md). Optional CAD conversion uses a separate environment with `usd-core`; do not install that provider into the Isaac Sim SDK environment.

See [Eka repair and validation](../source/isaaclab_hiwonder/EKA_V6_VALIDATION.md), [repaired CAD exports](../cad/eka_v6/README.md), and the [bounded simulation receipts](../validation/eka_v6-2026-10-28/summary.json).
