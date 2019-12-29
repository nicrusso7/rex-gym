clean:
	rm -r dist/
	rm -r *.egg-info

sdist: clean
	python setup.py sdist

deploy: sdist
	twine upload dist/*