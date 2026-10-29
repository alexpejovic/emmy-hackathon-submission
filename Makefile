PYTHON    ?= 3.12.12
VENV      ?= .venv
COMPILER  ?= compile.py
PTXGEN	  ?= ptxgen
EMMY      ?= $(VENV)/bin/emmy
TARGET    ?= sm_89
TORCH_IDX := https://download.pytorch.org/whl/cpu

.PHONY: envs clean lint compile-check submit-compiler submit-closed submit-ref generate-problems \
	install_devtools myptx

envs: $(VENV)

$(VENV):
	python -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/pip install emmy-ml==0.3.11 numpy requests torch

install_devtools:
	$(VENV)/bin/pip install torch ruff mypy "nvidia-cuda-nvcc-cu12>=12.9.86" # dev additions

clean:
	rm -rf $(VENV)

lint: $(VENV)
	$(VENV)/bin/ruff format
	$(VENV)/bin/ruff check --fix
	$(VENV)/bin/mypy $(PTXGEN) --strict
	$(VENV)/bin/mypy $(COMPILER) --strict

generate-problems: $(VENV)
	@rc=0; \
	$(VENV)/bin/python tools/generate.py --emmy $(EMMY) --target $(TARGET) --out problems tools/expressions_open.py || rc=1; \
	$(VENV)/bin/python tools/generate.py --emmy $(EMMY) --target $(TARGET) --out problems/closed tools/expressions_closed.py || rc=1; \
	exit $$rc

compile-check: $(VENV)
	@ok=0; failed=0; \
	for ir in problems/*/*.ir.json problems/closed/*/*.ir.json; do \
		if $(VENV)/bin/python $(COMPILER) "$$ir" > /dev/null 2>&1; then \
			echo "ok   $$ir"; ok=$$((ok + 1)); \
		else \
			echo "FAIL $$ir"; failed=$$((failed + 1)); \
		fi; \
	done; \
	echo "$$ok ok, $$failed failed"

myptx:
	PYTHON=$(VENV)/bin/python COMPILER=$(COMPILER) tools/gen_my_ptx.sh

submit-compiler: $(VENV)
	@ok=0; failed=0; \
	cd emmy-hackathon-2026 && for d in problems/*/; do \
	        p=$$(basename "$$d"); [ -f "$$d/meta.json" ] || continue; \
	        echo "== $$p"; \
	        if $(abspath $(VENV))/bin/python submit.py "$$p" --compiler $(abspath $(COMPILER)); then \
	                ok=$$((ok + 1)); \
	        else \
	                echo "FAIL $$p"; failed=$$((failed + 1)); \
	        fi; \
	done; \
	echo "$$ok ok, $$failed failed"

submit-closed: $(VENV)
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
