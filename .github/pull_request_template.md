## Summary

<!-- What does this pull request change, and why? Link the related issue, for example "Closes #12". -->

## Type of change

- [ ] Bug fix
- [ ] New feature or improvement
- [ ] Documentation
- [ ] CI, dependencies or tooling

## Checklist

- [ ] Python: `python -m pytest -v`, `ruff check python cpp/scripts` and `ruff format --check python cpp/scripts` pass (if `python/` or `cpp/scripts/` changed)
- [ ] C++: `make test` passes without new compiler warnings (if `cpp/` changed)
- [ ] Both implementations still produce the same coefficients (if the algorithm changed)
- [ ] Tests added or updated for this change
- [ ] Documentation updated (READMEs, wiki) if the usage or the behavior changed
- [ ] Changes to `cpp/third_party/` are listed in `cpp/third_party/README.md`
