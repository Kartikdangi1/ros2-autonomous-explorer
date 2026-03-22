from setuptools import setup

package_name = 'sensor_fusion_preprocess'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='Multi-sensor fusion preprocess node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_fusion_preprocess_node = '
            'sensor_fusion_preprocess.sensor_fusion_preprocess_node:main',
        ],
    },
)
