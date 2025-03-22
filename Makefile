.PHONY: ppo_train ppo_eval qlearn_train qlearn_eval

ppo_train:
	-@ulimit -n 4096
	python src/ppo_train.py

ppo_eval:
	python src/ppo_eval.py

qlearn_train:
	python src/qlearning_train.py

qlearn_eval:
	python src/qlearning_eval.py
