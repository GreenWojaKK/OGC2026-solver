PYTHON ?= /home/yoohs0422/ogcvenv/bin/python
CARGO ?= cargo
CXX ?= g++
INSTANCE ?= instances/train/prob_1.json
TL ?= 180

.PHONY: test run package native native-rust native-cpp

test:
	PYTHONPATH=solver $(PYTHON) -m unittest discover -s tests -p 'test_*.py'

run:
	$(PYTHON) tools/run_instance.py $(INSTANCE) --tl $(TL)

package:
	$(PYTHON) tools/package_submission.py

native: native-rust native-cpp

native-rust:
	$(CARGO) build --release --locked \
		--manifest-path native/rustcore/Cargo.toml
	cp native/rustcore/target/release/librustcore.so solver/rustcore.so

native-cpp:
	$(CXX) -O2 -shared -fPIC -ffp-contract=off -fno-math-errno \
		-Wall -Wextra -static-libstdc++ -static-libgcc \
		-o solver/geoexact.so native/geoexact/geoexact.cpp
