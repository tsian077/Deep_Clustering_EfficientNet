from setuptools import setup, find_packages

setup(name='DeWave',
      version='0.12',
      description='Single-channel blind source separation',
      long_description='Decomposing two overlapping speech signals that are \
      recoded in one channel and restoring signals for each speaker',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Build Tools',
      ],
      keywords=[
        'Blind source separation',
        'Single channel',
      ],
      url='https://github.com/chaodengusc/DeWave',
      author='Chao Deng',
      author_email='chaodengusc@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
        'tensorflow',
        'numpy',
        'scikit-learn',
        'librosa',
      ],
      entry_points={'console_scripts':[
        'dewave-clip=DeWave.cmddataprep:audioclips',
        'dewave-pack=DeWave.cmddatapack:packclips',
        'dewave-train=DeWave.cmdtrain:trainmodel',
        'dewave-infer=DeWave.cmdinfer:infer',
      ]},
      include_package_data=True,
      zip_safe=False)
