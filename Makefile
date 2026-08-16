# lmm — install guard hooks after a fresh clone.
# (core.hooksPath is repo-local, so a clone needs this one step.)
install-hooks:
	git config core.hooksPath .githooks
	chmod +x .githooks/pre-push
	@echo "[lmm] guard hooks active — every push runs the embedded selftest."

.PHONY: install-hooks
