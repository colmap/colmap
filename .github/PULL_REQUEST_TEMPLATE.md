## Breaking changes checklist

Please answer **both** questions below.  
Tick **exactly one** option (Yes/No) for each.  
If you select “Yes,” you must describe the impact and migration path.

---

### 1) Does this PR introduce breaking changes to **COLMAP public APIs**?
> A “breaking change” means external users of COLMAP (C++ APIs, CLI tools, flags, or documented interfaces) must modify their code, commands, or configurations when upgrading.

- [ ] No — no breaking changes to COLMAP APIs.
- [ ] Yes — this PR breaks COLMAP APIs.
  - **Impact & migration (COLMAP):** _Describe what breaks and how to migrate (e.g., changed APIs, removed flags, new parameters, compatibility notes)._

---

### 2) Does this PR introduce breaking changes to **pycolmap public APIs**?
> A “breaking change” means Python users of pycolmap must update imports, function/class signatures, return types/shapes, or behaviors when upgrading.

- [ ] No — no breaking changes to pycolmap APIs.
- [ ] Yes — this PR breaks pycolmap APIs.
  - **Impact & migration (pycolmap):** _Describe what breaks and how to migrate (e.g., renamed functions, changed argument order, new return types)._

---

### Additional context (optional)
- Motivation or rationale for the change:
