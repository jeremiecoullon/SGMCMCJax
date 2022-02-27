test:
	mypy sgmcmcjax/
	pytest -v sgmcmcjax tests --cov=sgmcmcjax --cov-report=html

