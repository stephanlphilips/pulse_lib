from setuptools import setup, find_packages


packages = find_packages()
print('packages: %s' % packages)


setup(name="pulse_lib",
	version="1.6.26",
	packages = find_packages(),
    python_requires=">=3.7",
    install_requires=[
            'si_prefix',
            'qcodes>=0.27.0',
            'numpy>=1.20.0',
            ],
    license='MIT',
    package_data={
        "pulse_lib": ["py.typed"],
        "pulse_lib.tests.keysight_data": ["*.hdf5"],
        },
	)

