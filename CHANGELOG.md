# Changelog

## Unreleased

### Fixed
- Binary task uses two named classes and integer labels instead of scalar regression.
- Training is explicit; bad inputs return nonzero exits.
- ZIP handling rejects unsafe paths and reads without extracting.

### Added
- Installable package, explicit CLI, local-only model paths and dotenv support.
- Meaningful offline regression tests and optional fake-backend adapter checks.
- Locked dependencies, quality hooks, public-only no-artifact CI and lightweight non-root CLI image.
- Honest verification, behavior-change and presentation documentation.

No trained-model quality, production readiness, deployment, release or license claim.
