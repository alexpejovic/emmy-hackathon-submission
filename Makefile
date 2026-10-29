PYTHON    ?= 3.12.12
VENV      ?= .venv
EMMY_VENV ?= .venv-emmy
COMPILER  ?= compile.py
PTXGEN	  ?= ptxgen
TORCH_IDX := https://download.pytorch.org/whl/cpu

.PHONY: envs clean-envs lint fmt compile-check ptxas-check myptx refptx submit-compiler \
	submit-closed submit-ref bench
envs: $(VENV)

$(VENV):
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install emmy-ml==0.3.11 numpy requests torch
	$(VENV)/bin/pip install torch ruff mypy "nvidia-cuda-nvcc-cu12>=12.9.86" # dev additions

clean:
	rm -rf $(VENV)

lint:
	$(VENV)/bin/ruff format
	$(VENV)/bin/ruff check --fix
	$(VENV)/bin/mypy $(PTXGEN) --strict
	$(VENV)/bin/mypy $(COMPILER) --strict

# Compile every problem IR with $(COMPILER); lenient (always exits 0) until all open problems
# compile. Running compile.py as a script keeps the emmy/ checkout from shadowing the package.
compile-check:
	@ok=0; failed=0; \
	for ir in problems/*/*.ir.json problems/closed/*/*.ir.json; do \
		if $(VENV)/bin/python $(COMPILER) "$$ir" > /dev/null 2>&1; then \
			echo "ok   $$ir"; ok=$$((ok + 1)); \
		else \
			echo "FAIL $$ir"; failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "$$ok ok, $$failed failed"

# Submit every open problem through $(COMPILER); keeps going on failure, prints a tally
submit-compiler:
	@ok=0; failed=0; \
	cd emmy-hackathon-2026 && for d in problems/*/; do \
	        p=$$(basename "$$d"); [ -f "$$d/meta.json" ] || continue; \
	        echo "== $$p"; \
	        if ../.venv/bin/python submit.py "$$p" --compiler $(abspath $(COMPILER)); then \
	                ok=$$((ok + 1)); \
	        else \
	                echo "FAIL $$p"; failed=$$((failed + 1)); \
	        fi; \
	done; \
	echo "$$ok ok, $$failed failed"

# Submit every closed problem (by folder path) through $(COMPILER)
submit-closed:
	@ok=0; failed=0; \
	for d in problems/closed/*/; do \
	        echo "== $$d"; \
	        if $(VENV)/bin/python emmy-hackathon-2026/submit.py "$$d" --compiler $(abspath $(COMPILER)); then \
	                ok=$$((ok + 1)); \
	        else \
	                echo "FAIL $$d"; failed=$$((failed + 1)); \
	        fi; \
	done; \
	echo "$$ok ok, $$failed failed"
