# ADR 002: Declarative Skills and a trusted runtime boundary

Status: Accepted

Agent Skills consist only of Pydantic-validated `skill.yaml` metadata and Markdown instructions. Manifests may reference registered Python Tools and Validators; they cannot carry executable code. Duplicate IDs, missing components, dependency cycles and undeclared side effects fail Registry initialization.
