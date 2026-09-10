# Track A, Task 1 Report: Extend Caspar-HIP smoke test to the solver path

**Status: BLOCKED**

## What was done

1. Located the real Caspar solver API via `symforce/caspar/examples/bal/gen_and_run.py`
   (a genuine, non-toy BAL bundle-adjustment example): `CasparLibrary`,
   `@caslib.add_factor`, `lib.SolverParams()`, `lib.GraphSolver(params,
   <Node>_num_max=..., <factor>_num_max=...)`, `solver.set_*_from_stacked_host`
   / `set_*_from_host`, `solver.solve(print_progress=True)`,
   `solver.get_*_nodes_to_stacked_host`.
2. Wrote `~/git/symforce-rocm/hip_solver_smoke.py`, adapting that example's
   real `fac_reprojection` factor to a small **synthetic**, noise-free,
   known-ground-truth problem (no BAL download needed): 4 cameras, 16 points,
   fully connected (64 `fac_reprojection` factors), pinhole calibration.
   Ground-truth pixel observations computed independently in NumPy from the
   same projection formula as the factor.
3. Hit and fixed a mechanical HIP compile gap never reached by the existing
   toy-kernel smoke test: `symforce/caspar/source/runtime/cuda_to_hip.h` was
   missing `cudaGetDeviceCount` (used only by `GraphSolver`'s host-side device
   query in `solver.cc`, not by the toy kernel). Added the 1:1
   `#define cudaGetDeviceCount hipGetDeviceCount` alias next to the existing
   `cudaGetDevice`/`cudaSetDevice` aliases. Enumerated the rest of the
   generated tree's `cuda*` symbol usage against the alias table to confirm
   this was the only gap.
4. Ran the test (after fixing the alias) inside the required ROCm container
   (`--device=/dev/kfd --device=/dev/dri --group-add 39 --group-add 105
   --security-opt label=disable -e HSA_OVERRIDE_GFX_VERSION=11.5.1`), reusing
   the existing `localhost/symforce-hip-installed:tmp` image (added `cmake`
   via `apt-get install` + `docker commit`, since it was missing from that
   saved image despite being required by `caslib.compile()`).
5. Build succeeded, load succeeded, solve ran without crashing — but **did
   not converge**. Ran a second control test (perturbation scaled down 50x,
   an easy/well-determined problem) specifically to rule out BA gauge freedom
   (7-DOF null space from having no anchored/fixed camera) as an innocent
   explanation for nonzero final error. Both runs show the same stall
   signature: cost drops for the first ~10-20 LM iterations then plateaus,
   `step_quality ≈ 0.000` on nearly every subsequent iteration, and the LM
   damping parameter (`diag`) grows unboundedly (doubling on almost every
   rejected step) instead of the optimizer finding a valid descent direction.
   This is the classic signature of an incorrect PCG/Gauss-Newton step, not
   gauge freedom or ill-conditioning (the control case was trivially easy and
   still stalled at nonzero cost instead of returning near the global
   minimum of 0).
6. Per this task's explicit instructions, did **not** attempt speculative
   fixes to symforce-rocm's own HIP solver runtime (`solver_tools.cu`,
   `SortIndices`, or the `caspar_hip::reduce_sum`/`labeled_partition`
   butterfly-reduction emulation in `cuda_to_hip.h`, which the completion
   plan itself flags in advance as a suspect construct for non-contiguous
   cooperative groups — precisely what PCG's Jacobian/gradient accumulation
   exercises).
7. Documented the full failure mode (numbers, both runs, diagnosis) under a
   `## BLOCKED` heading in `~/git/symforce-rocm/docs/rocm-integration.md`,
   and committed/pushed to `origin/hip-integration`.

## Key finding

Caspar's toy-kernel path (4 accessor types: `ReadShared`, `ReadUnique`,
`AddSharedSum`, `WriteIndexed`) is verified correct on HIP/gfx1151 (prior
work, `hip_smoke.py`). Caspar's actual bundle-adjustment **solver path**
(`GraphSolver`/PCG/`SortIndices`) builds and runs without crashing on
HIP/gfx1151, but **does not converge** — it consistently stalls at nonzero
cost with near-zero `step_quality` and runaway LM damping, on both a
moderately-perturbed and a near-trivial synthetic problem. This is a genuine
correctness gap in the HIP port of the solver runtime, most likely in the
PCG linear-solve path's shared-memory reduction/accumulation
(`caspar_hip::reduce_sum` / `labeled_partition` in `cuda_to_hip.h`, or
`solver_tools.cu`), not something an easy build-time compat-header patch can
fix.

## Recommendation

Track B (COLMAP wiring of Caspar-HIP into `colmap-rocm`) should **not**
proceed on the assumption that the HIP solver path produces correct
bundle-adjustment results, until this convergence bug is root-caused and
fixed upstream in `symforce-rocm`'s HIP port. This blocks Track B Task 4's
"Step 3: Write the Caspar-specific cooperative_groups compat header" premise
of building the header purely from a construct inventory — the deeper issue
is not a missing construct mapping but (most likely) a semantically wrong
one already present.

## Files touched

- `~/git/symforce-rocm/hip_solver_smoke.py` (new)
- `~/git/symforce-rocm/docs/rocm-integration.md` (new `## BLOCKED` entry)
- `~/git/symforce-rocm/symforce/caspar/source/runtime/cuda_to_hip.h` (added
  one alias — `cudaGetDeviceCount`)
- `~/git/symforce-rocm/.gitignore` (added `hip_solver_smoke_generated/`)

Commit: `symforce-rocm` `hip-integration` `42bf1a18..de81cce3`
(`de81cce3`), pushed to `origin/hip-integration`.
