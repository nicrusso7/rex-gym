clean:
	rm -rf dist/
	rm -rf *.egg-info
	find . | grep -E "\(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

sdist: clean
	python setup.py sdist

deploy: sdist
	twine upload dist/*