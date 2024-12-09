from setuptools import setup, find_packages


packages = find_packages()
print('packages: %s' % packages)


setup(name="pulse_lib",
	version="1.7.36",
	packages = find_packages(),
    python_requires=">=3.10",
    install_requires=[
            'qcodes >= 0.27.0',
            'numpy >= 1.24, < 2.0',
            'scipy',
            ],
    license='MIT',
    package_data={
        "pulse_lib": ["py.typed"],
        "pulse_lib.tests.keysight_data": ["*.hdf5"],
        },
	)

