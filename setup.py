from setuptools import setup, find_packages


packages = find_packages()
print('packages: %s' % packages)


setup(name="pulse_lib",
	version="1.3.5",
	packages = find_packages(),
    python_requires=">=3.7",
    install_requires=['si_prefix', 'qcodes>=0.27.0'],
    license='MIT',
	)

