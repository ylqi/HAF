from setuptools import setup, find_packages


setup(name='HAF',
      version='0.1.0',
      description='Open-source toolbox for Image-based Localization (Place Recognition)',
      author_email='yanliqi@westlake.edu.cn',
      url='https://github.com/YanLiqi/HAF',
      license='MIT',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Image Localization',
          'Image Retrieval',
          'Place Recognition'
      ])
