# One-time repo setup: install dependencies and configure git hooks
.PHONY: setup
setup:
	git config core.hooksPath .githooks
	make -C calculator install
	make -C eu_ai_act_news install
